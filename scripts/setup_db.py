# scripts/setup_db.py

import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from core.config import settings
from core.logger import setup_logger

logger = setup_logger("setup_db")

def download_knowledge_base():
    """
    Checks if the SQLite DB and Qdrant data already exist locally. If not, it downloads them from the specified Hugging Face repository.
     - It uses snapshot_download with allow_patterns to only download the necessary files, optimizing speed and storage.
     - If the files already exist, it logs a message and skips the download.
    """
    sqlite_path = settings.SQLITE_PATH
    qdrant_dir = settings.QDRANT_PATH

    if os.path.exists(sqlite_path) and os.path.isdir(qdrant_dir):
        logger.info(f"⚡ SQLite DB and Qdrant data already exist at {sqlite_path} and {qdrant_dir}. Skipping download.")
        return
    
    repo_id = settings.REPO_ID
    local_dir = settings.DATA_DIR
    
    logger.info(f"📥 Downloading DBs from HF Repo: {repo_id} to {local_dir}...")
    
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["corpus/*", "qdrant/*"],
            ignore_patterns=["build_cache/*", ".gitattributes"],
            max_workers=4
        )
        logger.info(f"✅ Download complete! Data is ready at: {download_path}")
        
    except HfHubHTTPError as e:
        logger.error(f"❌ HTTP Error during download: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during download: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    download_knowledge_base()