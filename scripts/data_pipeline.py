# scripts/data_pipeline.py

import json
import os
import re
import sqlite3

import numpy as np
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, OptimizersConfigDiff, PointStruct,
                                  ScalarQuantization, ScalarQuantizationConfig,
                                  ScalarType, SparseIndexParams, SparseVector,
                                  SparseVectorParams, VectorParams)
from tqdm import tqdm
from transformers import AutoTokenizer


class KnowledgeEngineBuilder:
    def __init__(self, base_dir="ke_store", dim=1024):
        self.base_dir = base_dir
        self.dim = dim
        
        print("Loading BGE-M3 Model and Tokenizer...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

        self.max_tokens = 384 
        self.overlap_count = 2 
        
        self._init_dirs()
        self._init_sqlite()  
        self._init_meta()
        self._init_qdrant()

    # ---------------------------
    # INIT & SETUP
    # ---------------------------
    def _init_dirs(self):
        for d in ["corpus", "qdrant", "build_cache/embeddings"]:
            os.makedirs(os.path.join(self.base_dir, d), exist_ok=True)

    def _init_qdrant(self):
        self.qdrant_path = f"{self.base_dir}/qdrant"
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
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
                    scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                ),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0) 
            )

    def _optimize_sqlite(self, conn):
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-2000000")

    def _init_sqlite(self):
        self.conn = sqlite3.connect(f"{self.base_dir}/corpus/corpus.sqlite")
        self._optimize_sqlite(self.conn)
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT, title TEXT, lang TEXT, url TEXT,
            wikidata_id TEXT, date_modified TEXT, full_text TEXT)
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER, chunk_index INTEGER, text TEXT,
            token_length INTEGER, section TEXT, lang TEXT)
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            span_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER, span_index INTEGER, text TEXT, char_length INTEGER)
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_spans_chunk_id ON spans(chunk_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang)")
        self.conn.commit()

    def _init_meta(self):
        self.meta_path = f"{self.base_dir}/corpus/meta.json"
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(doc_id) FROM documents")
        db_doc = cur.fetchone()[0] or 0
        cur.execute("SELECT MAX(chunk_id) FROM chunks")
        db_chunk = cur.fetchone()[0] or 0
        cur.execute("SELECT MAX(span_id) FROM spans")
        db_span = cur.fetchone()[0] or 0

        self.meta = {
            "last_doc_id": db_doc + 1,
            "last_chunk_id": db_chunk + 1,
            "last_span_id": db_span + 1
        }
        self._save_meta()

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=4)

    # ---------------------------
    # TEXT PROCESSING & INGESTION
    # ---------------------------
    def split_sentences(self, text):
        text = re.sub(r'[ \t]+', ' ', text) 
        pattern = r'(?<=[.!?。！？])(?<![Ar|Dr|Mr|Ms|St]\.)(?<![A-Z]\.)\s+'
        sentences = re.split(pattern, text)
        final_sentences = []
        for s in sentences:
            sub_parts = [p.strip() for p in s.split('\n') if p.strip()]
            final_sentences.extend(sub_parts)
        return [s for s in final_sentences if len(s) > 1]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def get_token_counts_batch(self, texts):
        if not texts: return []
        encodings = self.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        return [len(ids) for ids in encodings['input_ids']]

    def _split_monster_sentence(self, sentence):
        words = sentence.split(' ')
        sub_spans, current_sub, current_toks = [], [], 0

        for word in words:
            word_toks = self.count_tokens(word)
            if word_toks > self.max_tokens:
                if current_sub:
                    sub_spans.append(" ".join(current_sub))
                    current_sub, current_toks = [], 0
                half = len(word) // 2
                sub_spans.extend([word[:half], word[half:]])
                continue

            space_tok = 1 if current_sub else 0 
            if current_toks + word_toks + space_tok > self.max_tokens and current_sub:
                sub_spans.append(" ".join(current_sub))
                current_sub, current_toks = [word], word_toks
            else:
                current_sub.append(word)
                current_toks += word_toks + space_tok
                
        if current_sub: sub_spans.append(" ".join(current_sub))
        return sub_spans

    def chunk_text(self, text):
        raw_sentences = self.split_sentences(text)
        sentence_lengths = self.get_token_counts_batch(raw_sentences)
        
        refined_spans = []
        for s, length in zip(raw_sentences, sentence_lengths):
            if length > self.max_tokens: refined_spans.extend(self._split_monster_sentence(s))
            else: refined_spans.append(s)

        span_toks_list = self.get_token_counts_batch(refined_spans)
        chunks, current_spans, current_tokens = [], [], 0

        for span, span_toks in zip(refined_spans, span_toks_list):
            if current_tokens + span_toks > self.max_tokens and current_spans:
                chunk_text = " ".join(current_spans)
                chunks.append((chunk_text, self.count_tokens(chunk_text), list(current_spans)))
                
                actual_overlap = min(self.overlap_count, len(current_spans) - 1)
                if actual_overlap > 0:
                    current_spans = current_spans[-actual_overlap:]
                    current_tokens = self.count_tokens(" ".join(current_spans)) + 1 
                else:
                    current_spans, current_tokens = [], 0

            current_spans.append(span)
            current_tokens += span_toks + 1 

        if current_spans:
            chunk_text = " ".join(current_spans)
            chunks.append((chunk_text, self.count_tokens(chunk_text), list(current_spans)))
        return chunks

    def ingest(self, lang="ko", batch_size=32, limit=None):
        """
         - The dataset is read in a streaming manner to handle large corpora without memory issues.
         - Each document is processed to create chunks based on token limits, with an overlap strategy to ensure comprehensive coverage of the text.
            - The processed documents, chunks, and spans are stored in SQLite with appropriate indexing for efficient retrieval during search.
         """
        ds = load_dataset("HuggingFaceFW/finewiki", lang, split="train", streaming=True)
        cur = self.conn.cursor()
        count = 0
        batch_docs, batch_chunks, batch_spans = [], [], []

        for item in tqdm(ds, desc=f"Ingesting {lang}"):
            if limit and count >= limit: break
            doc_id = self.meta["last_doc_id"]
            batch_docs.append((doc_id, item["id"], item["title"], lang, item["url"], item.get("wikidata_id", ""), item.get("date_modified", ""), item["text"]))

            for c_idx, (chunk_text, token_len, span_list) in enumerate(self.chunk_text(item["text"])):
                chunk_id = self.meta["last_chunk_id"]
                batch_chunks.append((chunk_id, doc_id, c_idx, chunk_text, token_len, item["title"], lang))
                for s_idx, span_text in enumerate(span_list):
                    batch_spans.append((self.meta["last_span_id"], chunk_id, s_idx, span_text, len(span_text)))
                    self.meta["last_span_id"] += 1
                self.meta["last_chunk_id"] += 1
            self.meta["last_doc_id"] += 1
            count += 1

            if len(batch_docs) >= batch_size:
                self._commit_batch(cur, batch_docs, batch_chunks, batch_spans)
                batch_docs, batch_chunks, batch_spans = [], [], []
                if count % (batch_size * 10) == 0: self._save_meta()

        self._commit_batch(cur, batch_docs, batch_chunks, batch_spans)
        self.conn.commit()
        self.conn.execute("PRAGMA wal_checkpoint(FULL);")
        self._save_meta()

    def _commit_batch(self, cur, docs, chunks, spans):
        if not docs: return
        cur.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", docs)
        cur.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", chunks)
        cur.executemany("INSERT INTO spans VALUES (?,?,?,?,?)", spans)

    # ---------------------------
    # EMBED TO DISK 
    # ---------------------------
    def embed_corpus(self, lang="ko", batch_size=128, save_interval=100000):
        """
        Text is read in batches from SQLite, embeddings are generated using BGE-M3, and then saved to disk.  
         - Embedding generation is performed on the GPU, and data is saved to disk in fixed batches to manage memory.
         - Dense vectors are saved in NumPy's .npz format to ensure fast loading and low disk usage.
         - Sparse vectors are saved in JSONL format to provide flexibility and readability.
         - The saved embeddings are subsequently uploaded to Qdrant for use in searches.
         - This method is designed to reliably generate and save embeddings even on large-scale datasets.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, text FROM chunks WHERE lang=?", (lang,))
        rows = cur.fetchall()

        part_id = 0
        id_buffer = []
        dense_buffer = []
        sparse_buffer = []

        save_dir = f"{self.base_dir}/build_cache/embeddings"

        for i in tqdm(range(0, len(rows), batch_size), desc=f"1/2 GPU Embedding ({lang})"):
            batch = rows[i:i+batch_size]
            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]

            output = self.model.encode(
                texts, batch_size=len(texts), max_length=self.max_tokens, 
                return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
            
            id_buffer.extend(ids)
            dense_buffer.append(output['dense_vecs'])
            
            for sp_dict in output['lexical_weights']:
                sparse_buffer.append({str(k): float(v) for k, v in sp_dict.items()})

            # Save to disk when a certain number is reached (prevents memory explosion)
            if len(id_buffer) >= save_interval:
                self._save_embedding_part(save_dir, lang, part_id, id_buffer, dense_buffer, sparse_buffer)
                part_id += 1
                id_buffer, dense_buffer, sparse_buffer = [], [], []

        # Save the last remaining scraps
        self._save_embedding_part(save_dir, lang, part_id, id_buffer, dense_buffer, sparse_buffer)
        print(f"Embedding Generation Complete. Saved to {save_dir}")

    def _save_embedding_part(self, save_dir, lang, part_id, ids, dense_chunks, sparse_list):
        if not ids: return
        
        # Dense & IDs: High-speed storage as NumPy binaries
        np.savez(f"{save_dir}/ebd_{lang}_{part_id}.npz", 
                 ids=np.array(ids, dtype=np.int64), 
                 dense=np.vstack(dense_chunks))
        
        # Sparse: Save in JSONL format (one line at a time)
        with open(f"{save_dir}/sparse_{lang}_{part_id}.jsonl", 'w', encoding='utf-8') as f:
            for sp in sparse_list:
                f.write(json.dumps(sp) + '\n')

    # ---------------------------
    # BUILD QDRANT INDEX
    # ---------------------------
    def build_qdrant_index(self, lang="ko", batch_size=2000):
        """
        The generated embeddings are read from disk and uploaded to Qdrant in batches.  
         - This method reads the saved dense and sparse embeddings, constructs the appropriate data structures for Qdrant, and uploads them in batches to manage memory and ensure efficient indexing.
         - After all data is uploaded, it triggers Qdrant's indexing process to optimize search performance.
         - The use of batch uploads and on-disk storage allows this process to scale to large datasets without overwhelming system memory.
        """
        save_dir = f"{self.base_dir}/build_cache/embeddings"
        files = sorted([f for f in os.listdir(save_dir) if f.startswith(f"ebd_{lang}_") and f.endswith(".npz")])

        for file_name in files:
            part_id = file_name.split("_")[-1].split(".")[0]
            
            # 1. Load file and convert to Qdrant point structure
            npz_path = os.path.join(save_dir, file_name)
            sparse_path = os.path.join(save_dir, f"sparse_{lang}_{part_id}.jsonl")
            
            data = np.load(npz_path)
            ids = data['ids']
            dense_vecs = data['dense']
            
            with open(sparse_path, 'r', encoding='utf-8') as f:
                sparse_vecs = [json.loads(line) for line in f]

            points_batch = []
            
            # 2. Qdrant Upload Loop
            for i in tqdm(range(len(ids)), desc=f"2/2 Qdrant Uploading (Part {part_id})"):
                chunk_id = int(ids[i])
                sparse_dict = sparse_vecs[i]
                
                point = PointStruct(
                    id=chunk_id, 
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": SparseVector(
                            indices=[int(k) for k in sparse_dict.keys()],
                            values=list(sparse_dict.values())
                        )
                    },
                    payload={"chunk_id": chunk_id, "lang": lang}
                )
                points_batch.append(point)

                # Upload when stacked to batch size
                if len(points_batch) >= batch_size:
                    self.qdrant_client.upload_points(
                        collection_name=self.collection_name, 
                        points=points_batch
                    )
                    points_batch = []

            # Uploading leftover scraps
            if points_batch:
                self.qdrant_client.upload_points(
                    collection_name=self.collection_name, 
                    points=points_batch
                )

        print("Data upload complete. Enabling HNSW Indexing...")
        
        # 3. [Key] After all uploads are complete, re-enable indexing (default 20,000) to optimize the graph
        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=20000) 
        )
        print("Qdrant Indexing Complete!")

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


if __name__ == "__main__":
    builder = KnowledgeEngineBuilder()
    try:
        builder.ingest(lang="ko", batch_size=32, limit=10000)  # Process only 10,000 documents as an example
        builder.embed_corpus(lang="ko", batch_size=128, save_interval=5000)
        builder.build_qdrant_index(lang="ko", batch_size=2000)
    finally:
        builder.close()