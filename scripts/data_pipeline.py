# scripts/data_pipeline.py

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import orjson
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import HfApi, upload_folder
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, HnswConfigDiff,
                                  OptimizersConfigDiff, PayloadSchemaType,
                                  PointStruct, ScalarQuantization,
                                  ScalarQuantizationConfig, ScalarType,
                                  SparseIndexParams, SparseVector,
                                  SparseVectorParams, VectorParams)
from tqdm import tqdm
from transformers import AutoTokenizer


class KnowledgeEngineBuilder:
    def __init__(self, base_dir="ke_store", dim=1024, host="localhost", port=6333, grpc_port=6334):
        self.base_dir = base_dir
        self.dim = dim
        self.max_tokens = 512

        # Dynamic Overlap setting constants
        self.overlap_ratio = 0.12  # Use 12% of the chunk length as overlap (Sweet Spot)
        self.min_overlap = 30      # Minimum guaranteed overlap token count

        self.kb_dir = os.path.join(self.base_dir, "knowledge_base")
        self.artifacts_dir = os.path.join(self.base_dir, "artifacts/bge_m3_cache")

        print("Loading Initial Setup...")
        self._init_dirs()
        self._init_sqlite()
        self._init_qdrant(host, port, grpc_port)

        self.model = None
        self.tokenizer = None

        self.prefix_map = {
            "ko": "문서 제목",
            "en": "Document Title",
            "zh": "文档标题",
            "ja": "ドキュメントタイトル",
            "es": "Título del documento",
            "fr": "Titre du document",
            "de": "Dokumenttitel",
        }

    def _load_models(self):
        if self.model is None:
            print("Loading BGE-M3 Model and Tokenizer to GPU...")
            self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

    def _init_dirs(self):
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _init_qdrant(self, host, port, grpc_port):
        self.qdrant_client = QdrantClient(
            host=host, 
            port=port, 
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=300
        )
        self.collection_name = "knowledge_base"

        if not self.qdrant_client.collection_exists(self.collection_name):
            print(f"Creating Qdrant collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.dim, distance=Distance.COSINE, on_disk=True)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
                },
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=False
                    )
                ),
                hnsw_config=HnswConfigDiff(on_disk=True),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0)
            )

            # Index for metadata-based filtering search (e.g., language)
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name, field_name="lang", field_schema=PayloadSchemaType.KEYWORD
            )

    def _init_sqlite(self):
        self.conn = sqlite3.connect(f"{self.kb_dir}/corpus.sqlite", check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=-10000000;") # 10GB cache
        self.conn.execute("PRAGMA foreign_keys=ON;")

        cur = self.conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY,
            external_id TEXT, title TEXT, lang TEXT, url TEXT,
            wikidata_id TEXT, date_modified TEXT, full_text TEXT)
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id INTEGER, chunk_index INTEGER, text TEXT,
            token_length INTEGER, section TEXT, lang TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE)
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            span_id INTEGER PRIMARY KEY,
            chunk_id INTEGER, span_index INTEGER, text TEXT, char_length INTEGER,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE)
        """)
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_spans_chunk_id ON spans(chunk_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang)")
        self.conn.commit()

    # ---------------------------------------------------------------
    # PHASE 1: Sophisticated Semantic Chunking and SQLite Ingestion
    # ---------------------------------------------------------------
    def split_sentences(self, text, lang="ko"):
        """
        Global Multilingual Sentence Splitter
        1st: Physical separation based on line breaks (compatible with table and list data)
        2nd: Semantic separation based on punctuation
        """
        # 1st physical line break separation (remove empty strings)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Setting up 2nd Language-Specific Regular Expressions for Punctuation Separation
        if lang in ["ko", "zh", "ja"]:
            # CJK: Includes full-width characters, immediately separated
            pattern = r'(?<=[.!?。！？])\s*'
        else:
            # Global: Abbreviation Defense and Multilingual Period Support
            pattern = r'(?<=[.!?。！？।॥؟۔])(?<!\bMr\.)(?<!\bDr\.)(?<!\bMs\.)(?<!\bSt\.)(?<!\b[A-Z]\.)\s+'

        final_spans = []
        for line in lines:
            # Normalization of consecutive spaces and tabs within lines
            line = re.sub(r'[ \t]+', ' ', line)
            
            # Punctuation-based separation
            spans = [s.strip() for s in re.split(pattern, line) if len(s.strip()) > 0]
            final_spans.extend(spans)
                
        return final_spans

    def chunk_text(self, text, title="", lang="ko"):
        """
        Context-Aware Dynamic Overlap Chunker
        Injects the document's title at the top of each chunk to maximize BGE-M3 embedding context retention.
        """
        raw_sentences = self.split_sentences(text, lang)
        chunks = []
        
        # 1. Context Injection Format Settings Optimized for BGE-M3 (Fixed Prefix)
        prefix_label = self.prefix_map.get(lang, "Document Title")
        prefix = f"{prefix_label}: [{title}]\n" if title else ""
        prefix_toks = self.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        prefix_len = len(prefix_toks)
        
        # [Safety Mechanism] If the title itself is abnormally long and consumes all tokens, a forced cutoff is set to a maximum of 100 tokens.
        if prefix_len > 100:
            prefix_toks = prefix_toks[:100]
            prefix = self.tokenizer.decode(prefix_toks) + "...\n"
            prefix_len = len(prefix_toks)
            
        # 2. Calculation of the actual maximum number of tokens that can be inserted into the body (Span combinations + Overlap)
        eff_max_tokens = self.max_tokens - prefix_len
        
        current_spans = []
        current_tokens = 0 # Cumulative number of tokens in the body (excluding prefix)

        for span in raw_sentences:
            span_toks = len(self.tokenizer.encode(span, add_special_tokens=False))

            # ---------------------------------------------------------
            # Case 1: Monster Sentence (when a single Span exceeds eff_max_tokens)
            # ---------------------------------------------------------
            if span_toks > eff_max_tokens:
                # 1. If there is accumulated span, release it first.
                if current_spans:
                    chunk_body = " ".join(current_spans)
                    chunk_text_final = prefix + chunk_body
                    final_tokens = prefix_len + len(self.tokenizer.encode(chunk_body, add_special_tokens=False))
                    
                    chunks.append((chunk_text_final, final_tokens, list(current_spans)))

                    # Dynamic Overlap Calculation (Based on Emitted 'Body')
                    target_overlap = max(self.min_overlap, int(current_tokens * self.overlap_ratio))
                    prev_tokens = self.tokenizer.encode(chunk_body, add_special_tokens=False)
                    overlap_tokens = prev_tokens[-target_overlap:]
                    overlap_text = self.tokenizer.decode(overlap_tokens)
                    
                    current_spans = [overlap_text]

                # 2. Merging Overlap and Monster Sentences
                combined_text = " ".join(current_spans + [span]) if current_spans else span
                combined_tokens = self.tokenizer.encode(combined_text, add_special_tokens=False)

                # 3. Slicing Monster Sentences into eff_max_tokens (Sliding Window)
                i = 0
                while i + eff_max_tokens < len(combined_tokens):
                    slice_toks = combined_tokens[i : i + eff_max_tokens]
                    slice_text = self.tokenizer.decode(slice_toks)
                    
                    chunk_text_final = prefix + slice_text
                    # Configure db_spans to store only the text (slice_text)
                    chunks.append((chunk_text_final, prefix_len + len(slice_toks), [slice_text]))

                    # Overlap calculation when moving to the next window (Overlap inside monster sentences)
                    dyn_overlap = max(self.min_overlap, int(eff_max_tokens * self.overlap_ratio))
                    i += (eff_max_tokens - dyn_overlap)

                # 4. Save the remaining tail portion after the loop
                remainder_toks = combined_tokens[i:]
                if remainder_toks:
                    rem_text = self.tokenizer.decode(remainder_toks)
                    current_spans = [rem_text]
                    current_tokens = len(self.tokenizer.encode(rem_text, add_special_tokens=False))
                else:
                    current_spans = []
                    current_tokens = 0
                continue

            # ---------------------------------------------------------
            # Case 2: General Sentence (Accumulation of general sentences)
            # ---------------------------------------------------------
            # +1 is a fake calculation that takes into account spacing between sentences
            if current_tokens + span_toks + 1 <= eff_max_tokens:
                current_spans.append(span)
                current_tokens += span_toks + 1
            else:
                # 1. Release accumulated span upon overflow
                chunk_body = " ".join(current_spans)
                body_tokens = self.tokenizer.encode(chunk_body, add_special_tokens=False)
                
                chunk_text_final = prefix + chunk_body
                final_tokens = prefix_len + len(body_tokens)
                
                chunks.append((chunk_text_final, final_tokens, list(current_spans)))

                # 2. Dynamic Overlap Calculation (Based on Emitted 'Body')
                target_overlap = max(self.min_overlap, int(len(body_tokens) * self.overlap_ratio))
                overlap_tokens = body_tokens[-target_overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens)

                # 3. Start of new chunk (previous chunk overlap + current span)
                current_spans = [overlap_text, span]
                current_tokens = len(self.tokenizer.encode(" ".join(current_spans), add_special_tokens=False))

        # ---------------------------------------------------------
        # Handle remaining spans after loop termination
        # ---------------------------------------------------------
        if current_spans:
            chunk_body = " ".join(current_spans)
            chunk_text_final = prefix + chunk_body
            final_tokens = prefix_len + len(self.tokenizer.encode(chunk_body, add_special_tokens=False))
            chunks.append((chunk_text_final, final_tokens, list(current_spans)))

        return chunks

    def ingest_to_db(self, lang="ko", chunk_batch_size=10000, limit=None):
        self._load_models()
        ds = load_dataset("HuggingFaceFW/finewiki", lang, split="train", streaming=True)
        cur = self.conn.cursor()
        
        cur.execute("SELECT MAX(doc_id) FROM documents")
        next_doc_id = (cur.fetchone()[0] or 0) + 1
        
        cur.execute("SELECT MAX(chunk_id) FROM chunks")
        next_chunk_id = (cur.fetchone()[0] or 0) + 1
        
        cur.execute("SELECT MAX(span_id) FROM spans")
        next_span_id = (cur.fetchone()[0] or 0) + 1

        count = 0
        b_docs, b_chunks, b_spans = [], [], []

        for item in tqdm(ds, desc=f"1/3: Ingesting {lang}wiki to SQLite"):
            if limit and count >= limit: break
            
            doc_id = next_doc_id
            doc_title = item.get("title", "")

            b_docs.append((doc_id, item["id"], doc_title, lang, item.get("url", ""),
                           item.get("wikidata_id", ""), item.get("date_modified", ""), item["text"]))

            for c_idx, (c_text, c_len, span_list) in enumerate(self.chunk_text(item["text"], doc_title, lang)):
                chunk_id = next_chunk_id
                b_chunks.append((chunk_id, doc_id, c_idx, c_text, c_len, doc_title, lang))

                for s_idx, s_text in enumerate(span_list):
                    span_id = next_span_id
                    b_spans.append((span_id, chunk_id, s_idx, s_text, len(s_text)))
                    next_span_id += 1

                next_chunk_id += 1

            next_doc_id += 1
            count += 1

            if len(b_chunks) >= chunk_batch_size:
                self._commit(cur, b_docs, b_chunks, b_spans)
                b_docs, b_chunks, b_spans = [], [], []

        self._commit(cur, b_docs, b_chunks, b_spans)
        self.conn.commit()

    def _commit(self, cur, d, c, s):
        if d: cur.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", d)
        if c: cur.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", c)
        if s: cur.executemany("INSERT INTO spans VALUES (?,?,?,?,?)", s)

    # --------------------------------------------------------------
    # PHASE 2: GPU Embedding and Disk Caching (Full Resume Support)
    # --------------------------------------------------------------
    def embed_corpus(self, lang="ko", batch_size=1024):
        self._load_models()
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM chunks WHERE lang=?", (lang,))
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT chunk_id, text FROM chunks WHERE lang=? ORDER BY chunk_id ASC", (lang,))

        batch_idx = 0
        pbar = tqdm(total=total_chunks, desc="2/3 GPU Embedding to Disk")

        while True:
            batch = cur.fetchmany(batch_size)
            if not batch: break

            npz_path = f"{self.artifacts_dir}/chunk_{lang}_{batch_idx}.npz"
            jsonl_path = f"{self.artifacts_dir}/chunk_{lang}_{batch_idx}.jsonl"

            # Resume Defense Logic: Skip embedding if both .npz and .jsonl files for the batch already exist (Assumes that if .npz exists, .jsonl also exists, but double-checking for safety)
            if os.path.exists(npz_path) and os.path.exists(jsonl_path):
                batch_idx += 1
                pbar.update(len(batch))
                continue

            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]

            # GPU Batch Embedding with BGE-M3 (Dense + Sparse Extraction)
            output = self.model.encode(texts, batch_size=len(texts), max_length=self.max_tokens, return_dense=True, return_sparse=True)

            np.savez(npz_path, ids=np.array(ids), dense=output['dense_vecs'])

            # Ultra-fast serialization using orjson for sparse vectors (List of Dicts) to JSONL format
            with open(jsonl_path, 'wb') as f:
                for sp in output['lexical_weights']:
                    f.write(orjson.dumps({str(k): float(v) for k, v in sp.items()}) + b'\n')

            batch_idx += 1
            pbar.update(len(batch))

        pbar.close()

    # ----------------------------------------------------------------------
    # PHASE 3: Qdrant Server Parallel Upload and Indexing Finalized on Disk
    # ----------------------------------------------------------------------
    def upload_to_qdrant(self, lang="ko", parallel_workers=None):
        save_dir = self.artifacts_dir
        files = [f for f in os.listdir(save_dir) if f.startswith(f"chunk_{lang}_") and f.endswith(".npz")]

        if parallel_workers is None:
            num_cores = os.cpu_count() or 1
            parallel_workers = max(1, min(8, int(num_cores * 0.2)))  # Use up to 20% of CPU cores, capped at 8 workers
            
        def upload_worker(file_name):
            data = np.load(os.path.join(save_dir, file_name))
            ids, dense = data['ids'], data['dense']

            # Ultra-fast deserialization using orjson for sparse vectors (List of Dicts) from JSONL format
            with open(os.path.join(save_dir, file_name.replace(".npz", ".jsonl")), 'rb') as f:
                sparse = [orjson.loads(line) for line in f]

            points = []
            for j in range(len(ids)):
                points.append(PointStruct(
                    id=int(ids[j]),
                    vector={
                        "dense": dense[j].tolist(),
                        "sparse": SparseVector(indices=[int(k) for k in sparse[j].keys()],
                                               values=list(sparse[j].values()))
                    },
                    payload={"lang": lang, "chunk_id": int(ids[j])}
                ))

            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points,
                wait=False
            )

        print(f"3/3 Starting Qdrant parallel upload with {parallel_workers} workers...")
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            list(tqdm(executor.map(upload_worker, files), total=len(files), desc="Qdrant Upload"))

        print("Upload complete. Finalizing HNSW Index on Disk...")
        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=20000)
        )
        print("Pipeline Complete!")

    def close(self):
        """DB Connection Close Method for Safe Resource Management"""
        if hasattr(self, 'conn'):
            self.conn.close()
            print("SQLite connection closed.")

    def wait_for_indexing(self):
        """
        Wait until optimizer_status is 'ok' and there are no ongoing tasks  
        (indicating that indexing is complete and the collection is fully optimized on disk)
        """
        print("Waiting for Qdrant to finish indexing (HNSW Merging)...")
        while True:
            try:
                info = self.qdrant_client.get_collection(self.collection_name)

                if info.status == "green":
                    print("Indexing confirmed complete.")
                    break
            except Exception as e:
                print(f"Checking index status... (Error: {e})")
                print("Retrying in 10 seconds...")
            
            time.sleep(10)


    # Magic method to support Python's 'with' statement for automatic resource management
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Wait for indexing only when there are no exceptions (exc_type) (normal exit).
        if exc_type is None:
            self.wait_for_indexing()
        else:
            print(f"Pipeline failed with error, skipping index wait: {exc_val}")


def manage_qdrant_server(storage_path, http_port=6333, grpc_port=6334):
    """Helper function that manages the lifecycle of the Qdrant server"""
    abs_storage_path = os.path.abspath(storage_path)
    os.makedirs(abs_storage_path, exist_ok=True)

    # 1. Terminate existing processes (prevent port conflicts)
    subprocess.run(["pkill", "-9", "qdrant"], capture_output=True)
    
    # 2. Check for Binary Existence (Installation Guide)
    if not os.path.exists("./qdrant"):
        print("Error: 'qdrant' binary not found in current directory.")
        print("Please download it first: wget https://github.com/qdrant/qdrant/releases/download/v1.16.2/qdrant-x86_64-unknown-linux-gnu.tar.gz")
        sys.exit(1)

    print(f"Starting Qdrant server [Storage: {abs_storage_path}]...")
    env = os.environ.copy()
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_port)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_port)
    env["QDRANT__STORAGE__STORAGE_PATH"] = abs_storage_path

    log_file = open("qdrant_log.txt", "w")
    process = subprocess.Popen(
        ["./qdrant"], 
        env=env, 
        stdout=log_file, 
        stderr=log_file,
        preexec_fn=os.setpgrp
    )
    time.sleep(10) # Waiting for server initialization
    return process



if __name__ == "__main__":
    # ---1. CLI Argument Settings---
    parser = argparse.ArgumentParser(description="Knowledge Engine Data Pipeline Runner")
    parser.add_argument("--lang", type=str, default="ko", help="Language code (e.g., ko, en)")
    parser.add_argument("--chunk_batch_size", type=int, default=10000, help="Batch size for SQLite ingestion")
    parser.add_argument("--limit", type=int, default=50000, help="Ingestion document limit")
    parser.add_argument("--batch_size", type=int, default=1024, help="Embedding batch size")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for Qdrant upload")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after completion")
    parser.add_argument("--repo_id", type=str, default="user_id/repo", help="Hugging Face repository ID for upload (e.g., user_id/repo)")
    args = parser.parse_args()

    # --- 2. Environment Setup ---
    STORAGE_PATH = "./ke_store/qdrant_storage"
    
    # --- 3. Server Execution ---
    server_process = manage_qdrant_server(STORAGE_PATH)

    # --- 4. Pipeline Execution (Utilizing Context Manager) ---
    try:
        print(f"--- Starting Pipeline for language: {args.lang} ---")
        with KnowledgeEngineBuilder() as builder:
            builder.ingest_to_db(lang=args.lang, chunk_batch_size=args.chunk_batch_size, limit=args.limit)
            builder.embed_corpus(lang=args.lang, batch_size=args.batch_size)
            builder.upload_to_qdrant(lang=args.lang, parallel_workers=args.workers)
        
        print("--- Pipeline Execution Successful ---")

    except Exception as e:
        print(f"Critical Error during pipeline: {e}")
    
    finally:
        # --- 5. Graceful Shutdown ---
        print("Shutting down Qdrant server safely...")
        subprocess.run(["pkill", "-15", "qdrant"], check=False)
        time.sleep(5) # Waiting for data flush

    # --- 6. Hugging Face Upload (Optional) ---
    if args.upload:
        print("Uploading to Hugging Face Hub...")
        api = HfApi()
        upload_folder(
            repo_id=args.repo_id,
            folder_path="ke_store",
            repo_type="dataset"
        )
        print("Upload complete!")