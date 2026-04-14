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

> **High-performance Hybrid Search & Reranking Engine based on BGE-M3.** > An advanced knowledge retrieval API system that combines Dense/Sparse embeddings and optimizes precision with Cross-Encoders.


---

## 🚀 Key Features
* **Hybrid Search:** Seamlessly combines Dense & Sparse vector retrieval using Qdrant's Native Fusion API (BGE-M3).
* **Re-ranking:** Ensures top-tier precision by re-ordering search results via Cross-Encoder models.
* **Clean Architecture:** Highly modularized layers (API, Service, Storage, Models) for superior maintainability and scalability.
* **CI/CD Pipeline:** Fully automated deployment to Hugging Face Spaces using GitHub Actions and Docker.
* **Auto-Healing Data:** Robust startup logic via FastAPI `lifespan` that automatically synchronizes and validates the knowledge base.

---

## 🏗 Project Structure
This project follows the **Separation of Concerns (SoC)** principle to ensure the system remains extensible and testable.

```text
├── api/          # API Routing & Dependency Injection (DI)
├── core/         # Global Configuration (Pydantic Settings) & Exception Handling
├── models/       # AI Model Inference (Embedder, Reranker)
├── services/     # Business Logic & Search Pipeline Orchestration
├── storage/      # Infrastructure Layer (Qdrant, SQLite Clients)
├── scripts/      # Data Pipeline & Database Setup Scripts
├── templates/    # Demo UI (Jinja2 Templates)
└── main.py       # App Entry Point & Lifespan Management
```

---

## 🛠 Tech Stack
* **Framework:** FastAPI
* **Vector DB:** Qdrant (Local Path Mode)
* **RDBMS:** SQLite (Metadata & Corpus Storage)
* **ML Models:**
    * `BAAI/bge-m3` (Multi-functional Embedding)
    * `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
* **DevOps:** Docker, GitHub Actions, Hugging Face Hub

---

## 🔧 Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Hugging Face Access Token (Read/Write)

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
3. Run the application (The system will automatically download the necessary DB files on startup):
   ```bash
   python main.py
   # OR using uvicorn
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```

---

## 📡 API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Redirects to Search Demo UI |
| `POST` | `/api/v1/search/` | Executes JSON-based Hybrid Search |
| `GET` | `/api/v1/system/health/ping` | System health check (Heartbeat) |

---

## 💡 Architecture Insights
1.  **Dependency Injection:** Uses FastAPI `app.state` to manage singletons of AI models and DB clients, allowing for easy mocking during unit testing.
2.  **Hybrid RAG Pipeline:** Beyond simple vector similarity, this engine leverages Sparse embeddings for keyword-level precision, merged via Reciprocal Rank Fusion (RRF).
3.  **Deployment Ready:** Optimized for PaaS environments (like HF Spaces) through a containerized Docker setup and automated CI/CD.

---

## 📄 Documentation
For more detailed technical documentation, design decisions, and troubleshooting, please visit:
* [Personal Archive Link](https://minjae-portfolio.vercel.app/projects/ke)
* [Technical Design Blog](https://minjae-portfolio.vercel.app/blogs/ke-pd)


---