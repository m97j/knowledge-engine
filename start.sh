#!/bin/bash
# start.sh

echo "1. Downloading Knowledge Base Data (Syncing with Hugging Face)..."
# Download data first before executing FastAPI so that Qdrant can recognize it.
python scripts/setup_db.py

echo "2. Starting Qdrant Server in background..."
# Run in the background by explicitly specifying the repository path of Qdrant
export QDRANT__STORAGE__STORAGE_PATH="/app/data/vector_store/qdrant"
/usr/local/bin/qdrant &

# Wait until the Qdrant server is fully running before starting FastAPI
echo "Waiting for Qdrant to initialize..."
until curl -s http://localhost:6333/readyz > /dev/null; do
    echo "Qdrant is not ready yet. Retrying in 2 seconds..."
    sleep 2
done
echo "Qdrant is fully initialized!"

echo "3. Starting FastAPI Server..."
# Run Uvicorn in the foreground
uvicorn main:app --host 0.0.0.0 --port 7860