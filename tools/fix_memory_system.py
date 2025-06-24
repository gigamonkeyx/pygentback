#!/usr/bin/env python3
"""
Memory System Critical Fixes

Addresses the critical issues identified in the memory system analysis:
1. FAISS-GPU installation and configuration
2. Vector store interface compatibility 
3. Import path resolution
4. Database configuration setup
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n🔧 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success: {description}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()[:200]}...")
        else:
            print(f"   ❌ Failed: {description}")
            print(f"   Error: {result.stderr.strip()[:200]}...")
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def fix_faiss_installation():
    """Try to install FAISS with GPU support"""
    print("\n🚀 Fixing FAISS Installation")
    print("=" * 40)
    
    # Try conda first (recommended for FAISS-GPU)
    if run_command("conda --version", "Check conda availability"):
        print("   Attempting conda installation of FAISS-GPU...")
        success = run_command(
            "conda install -c pytorch -c nvidia faiss-gpu=1.7.4 -y",
            "Install FAISS-GPU via conda"
        )
        if success:
            return True
    
    # Fallback: Install CPU FAISS and plan GPU wrapper
    print("   Conda not available, installing CPU FAISS...")
    success = run_command(
        "pip install faiss-cpu",
        "Install FAISS-CPU as fallback"
    )
    
    if success:
        print("   ⚠️ Installed CPU-only FAISS. GPU acceleration will require custom implementation.")
        return True
    
    return False

def fix_cupy_cuda_path():
    """Set CUDA_PATH environment variable for CuPy"""
    print("\n🎯 Fixing CuPy CUDA Path")
    print("=" * 40)
    
    # Common CUDA installation paths on Windows
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
        "C:\\cuda",
        "C:\\CUDA"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            os.environ['CUDA_PATH'] = path
            print(f"   ✅ Set CUDA_PATH to: {path}")
            
            # Test CuPy with CUDA_PATH
            test_result = run_command(
                f'python -c "import os; os.environ[\'CUDA_PATH\']=\'{path}\'; import cupy as cp; print(\'CuPy working with CUDA_PATH\')"',
                "Test CuPy with CUDA_PATH"
            )
            
            if test_result:
                return True
    
    print("   ⚠️ CUDA installation not found in common paths")
    print("   💡 Manual fix: Set CUDA_PATH environment variable to your CUDA installation directory")
    return False

def create_missing_config_modules():
    """Create missing configuration modules"""
    print("\n⚙️ Creating Missing Configuration Modules")
    print("=" * 50)
    
    # Create config/database.py
    config_dir = Path("src/config")
    config_dir.mkdir(exist_ok=True)
    
    database_config = '''"""
Database Configuration Module

Provides database connection and configuration utilities.
"""

import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from typing import Optional

def get_database_url() -> str:
    """Get database URL from environment or use default"""
    return os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:password@localhost:5432/pygent_factory"
    )

def create_engine(database_url: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine"""
    from sqlalchemy import create_engine as sa_create_engine
    
    url = database_url or get_database_url()
    return sa_create_engine(url)

def get_session_maker(engine: Optional[Engine] = None):
    """Get SQLAlchemy session maker"""
    if engine is None:
        engine = create_engine()
    return sessionmaker(bind=engine)
'''
    
    database_file = config_dir / "database.py"
    database_file.write_text(database_config)
    print(f"   ✅ Created: {database_file}")
    
    # Create __init__.py if missing
    init_file = config_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Configuration package"""')
        print(f"   ✅ Created: {init_file}")
    
    return True

def fix_vector_store_compatibility():
    """Create compatibility fixes for vector store interface"""
    print("\n🔗 Fixing Vector Store Compatibility")
    print("=" * 45)
    
    # Create a compatibility wrapper
    vector_compat = '''"""
Vector Store Compatibility Layer

Fixes interface compatibility between legacy and modular vector storage.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CompatDocument:
    """Compatibility document wrapper"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class CompatResult:
    """Compatibility search result wrapper"""
    document: CompatDocument
    similarity_score: float
    distance: float = 0.0
    
    def __post_init__(self):
        if self.distance == 0.0:
            self.distance = 1.0 - self.similarity_score

class CompatVectorStore:
    """Compatibility vector store wrapper"""
    
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: Dict[str, CompatDocument] = {}
        self.embeddings: List[List[float]] = []
        self.doc_ids: List[str] = []
        
    async def add_documents(self, documents: List[CompatDocument]) -> None:
        """Add documents to store"""
        for doc in documents:
            self.documents[doc.id] = doc
            if doc.embedding:
                self.embeddings.append(doc.embedding)
                self.doc_ids.append(doc.id)
        
        logger.info(f"Added {len(documents)} documents to {self.collection_name}")
    
    async def similarity_search(self, 
                               query_embedding: List[float],
                               limit: int = 10,
                               similarity_threshold: float = 0.0,
                               metadata_filter: Optional[Dict[str, Any]] = None) -> List[CompatResult]:
        """Perform similarity search"""
        if not self.embeddings:
            return []
        
        # Simple cosine similarity (fallback implementation)
        import numpy as np
        
        query_np = np.array(query_embedding)
        query_norm = np.linalg.norm(query_np)
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            emb_np = np.array(emb)
            emb_norm = np.linalg.norm(emb_np)
            
            if query_norm > 0 and emb_norm > 0:
                similarity = np.dot(query_np, emb_np) / (query_norm * emb_norm)
            else:
                similarity = 0.0
            
            if similarity >= similarity_threshold:
                doc_id = self.doc_ids[i]
                doc = self.documents[doc_id]
                
                # Apply metadata filter if specified
                if metadata_filter:
                    match = all(
                        doc.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                similarities.append((similarity, doc))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, doc in similarities[:limit]:
            results.append(CompatResult(
                document=doc,
                similarity_score=similarity,
                distance=1.0 - similarity
            ))
        
        return results
    
    async def update_document(self, document: CompatDocument) -> None:
        """Update document in store"""
        if document.id in self.documents:
            self.documents[document.id] = document
            # Update embedding if changed
            if document.id in self.doc_ids:
                idx = self.doc_ids.index(document.id)
                if document.embedding:
                    self.embeddings[idx] = document.embedding

class CompatVectorStoreManager:
    """Compatibility vector store manager"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.stores: Dict[str, CompatVectorStore] = {}
        
    async def initialize(self):
        """Initialize manager"""
        logger.info("Compatibility vector store manager initialized")
    
    async def get_store(self, collection_name: str) -> CompatVectorStore:
        """Get or create vector store"""
        if collection_name not in self.stores:
            self.stores[collection_name] = CompatVectorStore(collection_name)
        return self.stores[collection_name]
'''
    
    compat_file = Path("src/storage/vector_compat.py")
    compat_file.write_text(vector_compat)
    print(f"   ✅ Created: {compat_file}")
    
    return True

def create_gpu_vector_operations():
    """Create GPU-accelerated vector operations using CuPy"""
    print("\n🎯 Creating GPU Vector Operations")
    print("=" * 40)
    
    gpu_ops = '''"""
GPU Vector Operations

Provides GPU-accelerated vector operations using CuPy as fallback
for FAISS-GPU when not available.
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available, falling back to CPU operations")

class GPUVectorOps:
    """GPU-accelerated vector operations"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            logger.info("GPU vector operations enabled")
        else:
            logger.info("Using CPU vector operations")
    
    def cosine_similarity_batch(self, 
                               query: np.ndarray, 
                               vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and batch of vectors"""
        
        if self.use_gpu and CUPY_AVAILABLE:
            return self._gpu_cosine_similarity(query, vectors)
        else:
            return self._cpu_cosine_similarity(query, vectors)
    
    def _gpu_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """GPU cosine similarity computation"""
        try:
            # Transfer to GPU
            query_gpu = cp.asarray(query)
            vectors_gpu = cp.asarray(vectors)
            
            # Normalize
            query_norm = cp.linalg.norm(query_gpu)
            vectors_norm = cp.linalg.norm(vectors_gpu, axis=1)
            
            # Compute dot products
            dots = cp.dot(vectors_gpu, query_gpu)
            
            # Compute similarities
            similarities = dots / (query_norm * vectors_norm + 1e-8)
            
            # Transfer back to CPU
            return cp.asnumpy(similarities)
            
        except Exception as e:
            logger.error(f"GPU similarity computation failed: {e}")
            return self._cpu_cosine_similarity(query, vectors)
    
    def _cpu_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """CPU cosine similarity computation"""
        query_norm = np.linalg.norm(query)
        vectors_norm = np.linalg.norm(vectors, axis=1)
        
        dots = np.dot(vectors, query)
        similarities = dots / (query_norm * vectors_norm + 1e-8)
        
        return similarities
    
    def top_k_indices(self, similarities: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top k indices and scores"""
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                similarities_gpu = cp.asarray(similarities)
                top_indices = cp.argpartition(similarities_gpu, -k)[-k:]
                top_indices = top_indices[cp.argsort(similarities_gpu[top_indices])][::-1]
                top_scores = similarities_gpu[top_indices]
                
                return cp.asnumpy(top_indices), cp.asnumpy(top_scores)
            except:
                pass
        
        # CPU fallback
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores

# Global instance
gpu_ops = GPUVectorOps()
'''
    
    gpu_file = Path("src/utils/gpu_vector_ops.py")
    gpu_file.write_text(gpu_ops)
    print(f"   ✅ Created: {gpu_file}")
    
    return True

def run_post_fix_test():
    """Run a quick test to verify fixes"""
    print("\n✅ Running Post-Fix Verification")
    print("=" * 40)
    
    # Test imports
    test_commands = [
        ("python -c \"import sys; sys.path.append('src'); from config.database import get_database_url; print('Database config: OK')\"", "Database config import"),
        ("python -c \"import sys; sys.path.append('src'); from storage.vector_compat import CompatVectorStore; print('Vector compat: OK')\"", "Vector compatibility import"),
        ("python -c \"import sys; sys.path.append('src'); from utils.gpu_vector_ops import gpu_ops; print('GPU ops: OK')\"", "GPU operations import"),
    ]
    
    for cmd, desc in test_commands:
        run_command(cmd, desc)
    
    # Test FAISS import
    run_command("python -c \"import faiss; print(f'FAISS version: {getattr(faiss, \'__version__\', \'unknown\')}')\"", "FAISS import test")

def main():
    """Run all critical fixes"""
    print("🔧 PyGent Factory Memory System Critical Fixes")
    print("=" * 50)
    print("Addressing critical issues identified in memory system analysis...")
    
    fixes = [
        ("FAISS Installation", fix_faiss_installation),
        ("CuPy CUDA Path", fix_cupy_cuda_path),
        ("Missing Config Modules", create_missing_config_modules),
        ("Vector Store Compatibility", fix_vector_store_compatibility),
        ("GPU Vector Operations", create_gpu_vector_operations),
    ]
    
    results = []
    for name, fix_func in fixes:
        try:
            success = fix_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Run verification
    run_post_fix_test()
    
    # Summary
    print("\n📊 Fix Results Summary")
    print("=" * 30)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    successful_fixes = sum(1 for _, success in results if success)
    print(f"\n🎯 {successful_fixes}/{len(results)} fixes completed successfully")
    
    if successful_fixes == len(results):
        print("🚀 All critical fixes applied! Memory system should be operational.")
    else:
        print("⚠️ Some fixes failed. Check the output above for manual resolution steps.")

if __name__ == "__main__":
    main()
