# src/scripts/build_knowledge_base.py

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

from src.orchestration.enhanced_document_acquisition import EnhancedDocumentAcquisition
from src.storage.vector.manager import VectorStoreManager
from src.utils.embedding import EmbeddingService
from src.config.settings import get_settings

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use a local, GPU-accelerated model for free, high-performance embeddings.
# This model is well-suited for the NVIDIA 3080.
# EMBEDDING_MODEL = "all-mpnet-base-v2" 
VECTOR_STORE_TYPE = "faiss" # FAISS is ideal for local, high-performance vector storage.
STORAGE_PATH = project_root / "data" / "documents"
VECTOR_DB_PATH = project_root / "data" / "vector_stores"

# A curated list of free, high-quality historical documents from the Internet Archive.
# This serves as the initial corpus for our knowledge base.
INITIAL_SOURCES = [
    {
        "url": "https://archive.org/download/personalmemoirso01gran/personalmemoirso01gran.pdf",
        "metadata": {
            "title": "Personal Memoirs of U. S. Grant, Vol. 1",
            "author": "Ulysses S. Grant",
            "source_type": "primary",
            "era": "American Civil War"
        }
    },
    {
        "url": "https://archive.org/download/personalmemoirso01sher/personalmemoirso01sher.pdf",
        "metadata": {
            "title": "Personal Memoirs of Gen. W. T. Sherman, Vol. 1",
            "author": "William T. Sherman",
            "source_type": "primary",
            "era": "American Civil War"
        }
    },
    {
        "url": "https://archive.org/download/upfromslaveryaut00wash/upfromslaveryaut00wash.pdf",
        "metadata": {
            "title": "Up from Slavery: An Autobiography",
            "author": "Booker T. Washington",
            "source_type": "primary",
            "era": "Reconstruction"
        }
    }
]

async def build_knowledge_base(provider: Optional[str] = None):
    """
    Main function to build the historical knowledge base.
    Initializes services, processes documents, and stores them in a vector database.
    """
    logger.info("--- Starting Knowledge Base Construction ---")
    
    # Ensure storage directories exist
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- Clean up previous runs ---
    logger.info(f"Cleaning up existing documents in {STORAGE_PATH} to ensure a fresh start.")
    for f in STORAGE_PATH.glob("*.pdf"):
        try:
            f.unlink()
            logger.info(f"Removed old file: {f}")
        except OSError as e:
            logger.error(f"Error removing file {f}: {e}")

    logger.info(f"Document storage path: {STORAGE_PATH}")
    logger.info(f"Vector DB path: {VECTOR_DB_PATH}")

    # 1. Initialize Core Services
    settings = get_settings()
    # settings.embedding_model = EMBEDDING_MODEL
    settings.vector_store = VECTOR_STORE_TYPE
    settings.faiss_db_path = str(VECTOR_DB_PATH)
    
    # Embedding service will automatically use the GPU (NVIDIA 3080) if available.
    embedding_service = EmbeddingService(settings)

    # Override default provider if specified via CLI
    if provider:
        available_providers = embedding_service.get_available_providers()
        if provider in available_providers:
            embedding_service.default_provider = provider
            logger.info(f"Using embedding provider specified via CLI: {provider}")
        else:
            logger.error(f"Invalid provider specified: '{provider}'. Available providers: {available_providers}")
            return

    # --- Preload Ollama Models ---
    # Preload models to avoid download delays during processing.
    # This is especially useful for large models.
    if settings.ai.OLLAMA_BASE_URL:
        logger.info("Preloading Ollama models...")
        # Models identified from previous `ollama list` command
        models_to_preload = ["qwen3:8b", "deepseek-r1:8b", "janus:latest"]
        if settings.ai.OLLAMA_EMBED_MODEL and settings.ai.OLLAMA_EMBED_MODEL not in models_to_preload:
            models_to_preload.append(settings.ai.OLLAMA_EMBED_MODEL)
        
        await embedding_service.preload_ollama_models(models_to_preload)
        logger.info("Ollama model preloading complete.")

    vector_manager = VectorStoreManager(settings, embedding_service)
    
    acquisition_service = EnhancedDocumentAcquisition(
        vector_manager=vector_manager,
        embedding_service=embedding_service,
        storage_path=str(STORAGE_PATH)
    )
    
    logger.info(f"Initialized services with Embedding Provider: '{embedding_service.default_provider}' and Vector Store: '{VECTOR_STORE_TYPE}'")

    # 2. Process Documents Sequentially for Robustness
    logger.info(f"Starting processing for {len(INITIAL_SOURCES)} initial documents.")
    
    success_count = 0
    failure_count = 0

    for source in INITIAL_SOURCES:
        try:
            logger.info(f"--- Processing URL: {source['url']} ---")
            # Add a timeout to prevent the script from hanging on a single document.
            task = acquisition_service.acquire_and_process_document(source["url"], source["metadata"])
            result = await asyncio.wait_for(task, timeout=300.0) # 5-minute timeout
            
            if result.get("success"):
                success_count += 1
                logger.info(f"✅ Successfully processed: {result['document_id']} ({result.get('file_path')})")
                logger.info(f"   - Text Quality: {result['text_quality']['quality']}")
                logger.info(f"   - Vector Chunks Stored: {result['vector_chunks']}")
            else:
                failure_count += 1
                logger.error(f"❌ Failed to process document. URL: {source['url']}, Error: {result.get('error')}")

        except asyncio.TimeoutError:
            failure_count += 1
            logger.error(f"❌ Timeout occurred while processing URL {source['url']}. The document took too long and was skipped.")
        except Exception as e:
            failure_count += 1
            logger.error(f"❌ An unexpected and critical error occurred for URL {source['url']}: {e}", exc_info=True)

    # 3. Report Results
    logger.info("--- Knowledge Base Construction Complete ---")
    
    logger.info("--- Final Summary ---")
    logger.info(f"Total Documents Attempted: {len(INITIAL_SOURCES)}")
    logger.info(f"Successful Ingestions: {success_count}")
    logger.info(f"Failed Ingestions: {failure_count}")
    
    final_stats = acquisition_service.get_processing_stats()
    logger.info(f"Detailed Stats: {final_stats}")
    logger.info("Knowledge base build process finished.")

if __name__ == "__main__":
    # This allows running the script directly.
    # Ensure you have the necessary packages installed:
    # pip install requests pymupdf sentence-transformers faiss-gpu
    
    # Note: 'faiss-gpu' is recommended for NVIDIA hardware. 
    # If you have issues, 'faiss-cpu' is a fallback.
    
    # To run this script from the project root:
    # python -m src.scripts.build_knowledge_base --provider ollama
    
    parser = argparse.ArgumentParser(
        description="Build the knowledge base by processing documents and generating embeddings."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Specify the embedding provider to use (e.g., 'ollama', 'openrouter', 'sentence_transformer', 'openai'). "
             "If not provided, the default from settings will be used."
    )
    args = parser.parse_args()
    
    asyncio.run(build_knowledge_base(provider=args.provider))
