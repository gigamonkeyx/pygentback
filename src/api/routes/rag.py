"""
RAG (Retrieval-Augmented Generation) API Routes

This module provides REST API endpoints for RAG system operations
including document processing, retrieval, and knowledge management.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from ...rag.retrieval_system import RetrievalSystem, RetrievalQuery, RetrievalStrategy
from ...rag.document_processor import DocumentProcessor
from ...security.auth import get_current_user, require_rag_read, require_rag_write, User


logger = logging.getLogger(__name__)

router = APIRouter()

# Global retrieval system instance (will be set by main.py)
_retrieval_system: Optional[RetrievalSystem] = None

def set_retrieval_system(system: RetrievalSystem):
    """Set the global retrieval system instance"""
    global _retrieval_system
    _retrieval_system = system

def get_retrieval_system() -> RetrievalSystem:
    """Get the retrieval system dependency"""
    if _retrieval_system is None:
        raise HTTPException(status_code=500, detail="Retrieval system not initialized")
    return _retrieval_system


# Request/Response models
class ProcessTextRequest(BaseModel):
    content: str
    title: str
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class RetrievalRequest(BaseModel):
    query: str
    strategy: str = "semantic"
    max_results: int = 10
    similarity_threshold: float = 0.7
    filters: Dict[str, Any] = {}
    context: Optional[str] = None
    agent_id: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    document_type: str
    source_path: Optional[str]
    source_url: Optional[str]
    chunk_count: int
    processed_at: datetime
    metadata: Dict[str, Any]


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_write)
):
    """Upload and process a document file"""
    try:
        # Parse metadata
        import json
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document
            from ...config.settings import get_settings
            settings = get_settings()
            document_processor = DocumentProcessor(settings)
            
            processed_doc = await document_processor.process_file(
                file_path=temp_file_path,
                title=title or file.filename,
                metadata={
                    "uploaded_by": current_user.username,
                    "original_filename": file.filename,
                    "file_size": len(content),
                    **metadata_dict
                }
            )
            
            # Add to retrieval system
            success = await retrieval_system.add_document(processed_doc)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to add document to retrieval system")
            
            return DocumentResponse(
                id=processed_doc.id,
                title=processed_doc.title,
                document_type=processed_doc.document_type.value,
                source_path=processed_doc.source_path,
                source_url=processed_doc.source_url,
                chunk_count=processed_doc.get_chunk_count(),
                processed_at=processed_doc.processed_at,
                metadata=processed_doc.metadata
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.post("/documents/text")
async def process_text(
    request: ProcessTextRequest,
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_write)
):
    """Process text content directly"""
    try:
        # Process text
        from ...config.settings import get_settings
        settings = get_settings()
        document_processor = DocumentProcessor(settings)
        
        processed_doc = await document_processor.process_text(
            content=request.content,
            title=request.title,
            source_url=request.source_url,
            metadata={
                "processed_by": current_user.username,
                **request.metadata
            }
        )
        
        # Add to retrieval system
        success = await retrieval_system.add_document(processed_doc)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to retrieval system")
        
        return DocumentResponse(
            id=processed_doc.id,
            title=processed_doc.title,
            document_type=processed_doc.document_type.value,
            source_path=processed_doc.source_path,
            source_url=processed_doc.source_url,
            chunk_count=processed_doc.get_chunk_count(),
            processed_at=processed_doc.processed_at,
            metadata=processed_doc.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@router.post("/retrieve")
async def retrieve_documents(
    request: RetrievalRequest,
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_read)
):
    """Retrieve relevant documents for a query"""
    try:
        # Parse strategy
        try:
            strategy = RetrievalStrategy(request.strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        # Create retrieval query
        query = RetrievalQuery(
            text=request.query,
            strategy=strategy,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters,
            context=request.context,
            agent_id=request.agent_id
        )
        
        # Perform retrieval
        results = await retrieval_system.retrieve(query)
        
        return {
            "query": request.query,
            "strategy": request.strategy,
            "total_results": len(results),
            "results": [
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "title": result.title,
                    "content": result.content,
                    "similarity_score": result.similarity_score,
                    "relevance_score": result.relevance_score,
                    "source_path": result.source_path,
                    "source_url": result.source_url,
                    "chunk_index": result.chunk_index,
                    "metadata": result.metadata,
                    "context_snippet": result.get_context_snippet()
                }
                for result in results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection: str = "documents",
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_write)
):
    """Delete a document from the retrieval system"""
    try:
        success = await retrieval_system.remove_document(document_id, collection)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/documents/{document_id}/similar")
async def find_similar_documents(
    document_id: str,
    limit: int = 5,
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_read)
):
    """Find documents similar to a given document"""
    try:
        results = await retrieval_system.search_similar_documents(document_id, limit)
        
        return {
            "document_id": document_id,
            "similar_documents": [
                {
                    "document_id": result.document_id,
                    "title": result.title,
                    "similarity_score": result.similarity_score,
                    "relevance_score": result.relevance_score,
                    "context_snippet": result.get_context_snippet()
                }
                for result in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar documents for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find similar documents: {str(e)}")


@router.get("/stats")
async def get_rag_stats(
    retrieval_system: RetrievalSystem = Depends(get_retrieval_system),
    current_user: User = Depends(require_rag_read)
):
    """Get RAG system statistics"""
    try:
        stats = await retrieval_system.get_retrieval_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")


@router.get("/strategies")
async def get_retrieval_strategies(
    current_user: User = Depends(require_rag_read)
):
    """Get available retrieval strategies"""
    return {
        "strategies": [strategy.value for strategy in RetrievalStrategy],
        "descriptions": {
            "semantic": "Pure semantic similarity search",
            "hybrid": "Combines semantic and keyword matching",
            "contextual": "Context-aware retrieval with agent memory",
            "adaptive": "Automatically selects best strategy based on query"
        }
    }


@router.get("/supported-formats")
async def get_supported_formats(
    current_user: User = Depends(require_rag_read)
):
    """Get supported document formats"""
    from ...config.settings import get_settings
    settings = get_settings()
    document_processor = DocumentProcessor(settings)
    
    return {
        "supported_extensions": document_processor.get_supported_extensions(),
        "document_types": [
            "text", "markdown", "pdf", "docx", "html", 
            "json", "yaml", "code"
        ]
    }
