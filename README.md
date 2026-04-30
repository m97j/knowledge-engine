---
title: Knowledge Engine
emoji: 🔍
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
license: apache-2.0
pinned: false
---

# 🔍 Knowledge Engine

[![Spaces](https://img.shields.io/badge/Demo-Spaces-FF9D00?logo=huggingface)](https://huggingface.co/spaces/m97j/knowledge-engine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3100/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> **High-performance Hybrid Search & Reranking Engine based on BGE-M3.** > An advanced knowledge retrieval API system designed for Agentic AI, combining Dense/Sparse embeddings and optimizing precision with Cross-Encoders.

---

## 🚀 Key Features
* **Hybrid Search (RRF):** Seamlessly combines Dense & Sparse vector retrieval using Qdrant's Native Fusion API (BGE-M3).
* **Cross-Encoder Re-ranking:** Ensures top-tier precision by re-ordering search results contextually via `bge-reranker-v2-m3`.
* **Agent-Ready Output:** Natively provides XML-tagged context blocks optimized for immediate injection into LLMs and Agentic workflows.
* **Auto-Healing & Sync:** Robust startup logic via FastAPI `lifespan` that automatically pulls pre-processed knowledge bases from Hugging Face Datasets and synchronizes them.
* **Clean Architecture:** Highly modularized layers (API, Service, Storage, Models) using Dependency Injection for superior maintainability.

---

## 🏗 Project Structure
Follows the **Separation of Concerns (SoC)** principle to ensure the system remains extensible and testable.

```text
├── api/          # API Routing & Schema Definitions
├── core/         # Global Configuration (Pydantic V2) & Exception Handling
├── models/       # AI Model Inference (Embedder, Reranker)
├── services/     # Business Logic & Search Pipeline Orchestration
├── storage/      # Infrastructure Layer (Qdrant, SQLite Clients)
├── scripts/      # Data Pipeline & HF Dataset Sync Scripts
├── templates/    # Demo UI (Jinja2 Templates)
└── main.py       # App Entry Point & Lifespan Management
```

---

## 🛠 Tech Stack
* **Framework:** FastAPI
* **Vector DB:** Qdrant (Server Mode)
* **RDBMS:** SQLite (Metadata & Corpus Storage)
* **ML Models:**
    * [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (Dense + Sparse Embedding)
    * [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) (Cross-Encoder)
* **DevOps:** Docker, GitHub Actions, Hugging Face Hub (Spaces & Datasets)
* **Corpus:** [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)(Currently consists only of kowiki; enwiki, eswiki, etc. to be added later)

---

## 🔧 Installation & Setup

### Prerequisites
* Python 3.10+
* Hugging Face Access Token (For initial setup/updates)

### Running Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/m97j/knowledge-engine.git](https://github.com/m97j/knowledge-engine.git)
   cd knowledge-engine
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   *(The system will automatically download the pre-built SQLite and Qdrant DB files from HF Datasets on startup via `scripts/setup_db.py`)*
   ```bash
   python main.py
   # OR
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```

### Preprocessing Pipeline (Optional)
If you want to build the knowledge base from scratch:
```bash
# 1. Download qdrant binary (Linux x86_64)
wget [https://github.com/qdrant/qdrant/releases/download/v1.16.2/qdrant-x86_64-unknown-linux-gnu.tar.gz](https://github.com/qdrant/qdrant/releases/download/v1.16.2/qdrant-x86_64-unknown-linux-gnu.tar.gz)
tar -xvf qdrant-x86_64-unknown-linux-gnu.tar.gz
chmod +x qdrant

# 2. Execute Pipeline
python scripts/data_pipeline.py --lang en --chunk_batch_size 10000 --limit 50000 --batch_size 1024 --workers 4 --upload --repo_id user/id
```

---

## 📡 API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Redirects to Search Demo UI |
| `POST` | `/api/v1/search/` | Executes JSON-based Hybrid Search (Returns structured JSON & LLM context) |
| `GET` | `/api/v1/system/health/ping` | System health check (Heartbeat) |

---

## 💡 Architecture Insights
1.  **O(1) Metadata Mapping:** By storing massive text payloads in SQLite and only vectors/IDs in Qdrant, we achieve extremely low latency during the reranking preparation phase.
2.  **Zero-Downtime Deployment:** Optimized for PaaS environments (like HF Spaces) through a containerized Docker setup and a custom `start.sh` that ensures DB readiness before FastAPI starts.

---

## 📄 Documentation
For more detailed technical documentation and design decisions:
* [Personal Archive Link](https://minjae-portfolio.vercel.app/projects/ke)
* [Technical Design Blog](https://minjae-portfolio.vercel.app/blogs/ke-pd)


---

