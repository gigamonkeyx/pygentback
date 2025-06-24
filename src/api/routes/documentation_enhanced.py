"""
Enhanced Documentation API with Persistent Backend Storage

This module provides comprehensive document management with:
- Persistent backend storage with full document lifecycle
- Session-based frontend temporary storage (cleared on logout)
- Document links for recall from backend
- Integration with research agent and document creation systems
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import markdown
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# Document models for persistent storage
class DocumentMetadata(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    category: str = Field(..., description="Document category")
    author: Optional[str] = Field(None, description="Document author/creator")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    file_path: str = Field(..., description="Backend file path")
    size_bytes: int = Field(0, description="Document size in bytes")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    is_generated: bool = Field(False, description="Whether document was AI-generated")
    research_session_id: Optional[str] = Field(None, description="Associated research session")
    access_count: int = Field(0, description="Number of times accessed")
    status: str = Field("active", description="Document status: active, archived, draft")

class SessionDocument(BaseModel):
    """Temporary document for frontend session (cleared on logout)"""
    session_id: str = Field(..., description="Frontend session identifier")
    document_id: str = Field(..., description="Reference to backend document")
    title: str = Field(..., description="Document title for display")
    preview_content: str = Field("", description="Truncated content preview")
    backend_link: str = Field(..., description="Link to retrieve full document from backend")
    cached_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)

class DocumentCreateRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field("General", description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    research_session_id: Optional[str] = Field(None, description="Associated research session")

class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Updated document title")
    content: Optional[str] = Field(None, description="Updated document content")
    category: Optional[str] = Field(None, description="Updated document category")
    tags: Optional[List[str]] = Field(None, description="Updated document tags")
    change_summary: Optional[str] = Field(None, description="Summary of changes made")

# In-memory storage for session documents (cleared on logout)
session_documents: Dict[str, List[SessionDocument]] = {}

# Persistent backend storage (simulated - would use database in production)
persistent_documents: Dict[str, DocumentMetadata] = {}
document_contents: Dict[str, str] = {}

def get_session_id(request: Request) -> str:
    """Extract session ID from request headers or generate new one"""
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

def clean_expired_sessions():
    """Clean up expired session documents"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, documents in session_documents.items():
        # Remove documents older than 24 hours or not accessed in 2 hours
        valid_documents = []
        for doc in documents:
            age = current_time - doc.cached_at
            idle_time = current_time - doc.last_accessed
            if age < timedelta(hours=24) and idle_time < timedelta(hours=2):
                valid_documents.append(doc)
        
        if valid_documents:
            session_documents[session_id] = valid_documents
        else:
            expired_sessions.append(session_id)
    
    # Remove completely empty sessions
    for session_id in expired_sessions:
        del session_documents[session_id]

@router.post("/create")
async def create_document(
    request: DocumentCreateRequest,
    session_id: str = Depends(get_session_id)
):
    """Create a new document with persistent backend storage"""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save to persistent backend storage
        file_path = f"documents/{doc_id}.md"
        document_contents[doc_id] = request.content
        
        metadata = DocumentMetadata(
            id=doc_id,
            title=request.title,
            category=request.category,
            file_path=file_path,
            size_bytes=len(request.content.encode('utf-8')),
            tags=request.tags,
            research_session_id=request.research_session_id,
            is_generated=bool(request.research_session_id)  # Mark as generated if from research
        )
        persistent_documents[doc_id] = metadata
        
        # Create temporary session reference (no content stored)
        preview_content = request.content[:200] + "..." if len(request.content) > 200 else request.content
        backend_link = f"/api/v1/documentation/documents/{doc_id}"
        
        session_doc = SessionDocument(
            session_id=session_id,
            document_id=doc_id,
            title=request.title,
            preview_content=preview_content,
            backend_link=backend_link
        )
        
        # Add to session documents
        if session_id not in session_documents:
            session_documents[session_id] = []
        session_documents[session_id].append(session_doc)
        
        logger.info(f"Created document {doc_id} for session {session_id}")
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'document_id': doc_id,
                'backend_link': backend_link,
                'session_reference': session_doc.dict(),
                'metadata': metadata.dict()
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document")

@router.get("/session")
async def get_session_documents(session_id: str = Depends(get_session_id)):
    """Get temporary session documents (links only, cleared on logout)"""
    try:
        clean_expired_sessions()  # Clean up old sessions
        
        session_docs = session_documents.get(session_id, [])
        
        # Update last accessed time
        current_time = datetime.now()
        for doc in session_docs:
            doc.last_accessed = current_time
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'session_id': session_id,
                'documents': [doc.dict() for doc in session_docs],
                'total': len(session_docs),
                'note': 'Session documents are temporary and cleared on logout'
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting session documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session documents")

@router.post("/session/clear")
async def clear_session_documents(session_id: str = Depends(get_session_id)):
    """Clear session documents (simulates logout)"""
    try:
        if session_id in session_documents:
            del session_documents[session_id]
            logger.info(f"Cleared session documents for session {session_id}")
        
        return JSONResponse(content={
            'status': 'success',
            'message': 'Session documents cleared (simulating logout)'
        })
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@router.get("/documents/{document_id}")
async def get_document_from_backend(
    document_id: str,
    session_id: str = Depends(get_session_id)
):
    """Retrieve full document from persistent backend storage"""
    try:
        if document_id not in persistent_documents:
            raise HTTPException(status_code=404, detail="Document not found in backend")
        
        metadata = persistent_documents[document_id]
        content = document_contents.get(document_id, "")
        
        # Update access count
        metadata.access_count += 1
        metadata.updated_at = datetime.now()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            content, 
            extensions=['toc', 'codehilite', 'fenced_code', 'tables']
        )
        
        logger.info(f"Retrieved document {document_id} from backend for session {session_id}")
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'metadata': metadata.dict(),
                'content': content,
                'html_content': html_content,
                'storage_location': 'backend_persistent',
                'access_info': {
                    'access_count': metadata.access_count,
                    'last_updated': metadata.updated_at.isoformat()
                }
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document from backend")

@router.put("/documents/{document_id}")
async def update_document_in_backend(
    document_id: str,
    request: DocumentUpdateRequest,
    session_id: str = Depends(get_session_id)
):
    """Update document in persistent backend storage"""
    try:
        if document_id not in persistent_documents:
            raise HTTPException(status_code=404, detail="Document not found in backend")
        
        metadata = persistent_documents[document_id]
        
        # Update metadata
        if request.title:
            metadata.title = request.title
        if request.category:
            metadata.category = request.category
        if request.tags is not None:
            metadata.tags = request.tags
        
        metadata.updated_at = datetime.now()
        
        # Update content if provided
        if request.content:
            document_contents[document_id] = request.content
            metadata.size_bytes = len(request.content.encode('utf-8'))
        
        # Update session reference if it exists
        if session_id in session_documents:
            for session_doc in session_documents[session_id]:
                if session_doc.document_id == document_id:
                    if request.title:
                        session_doc.title = request.title
                    if request.content:
                        preview = request.content[:200] + "..." if len(request.content) > 200 else request.content
                        session_doc.preview_content = preview
                    session_doc.last_accessed = datetime.now()
                    break
        
        logger.info(f"Updated document {document_id} in backend")
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'metadata': metadata.dict(),
                'change_summary': request.change_summary or "Document updated",
                'updated_at': metadata.updated_at.isoformat()
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update document")

@router.delete("/documents/{document_id}")
async def delete_document_from_backend(
    document_id: str,
    session_id: str = Depends(get_session_id)
):
    """Delete document from persistent backend storage"""
    try:
        if document_id not in persistent_documents:
            raise HTTPException(status_code=404, detail="Document not found in backend")
        
        # Remove from persistent storage
        del persistent_documents[document_id]
        if document_id in document_contents:
            del document_contents[document_id]
        
        # Remove from all session references
        for sid, docs in session_documents.items():
            session_documents[sid] = [d for d in docs if d.document_id != document_id]
        
        logger.info(f"Deleted document {document_id} from backend")
        
        return JSONResponse(content={
            'status': 'success',
            'message': 'Document deleted from backend and all session references'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.get("/backend/documents")
async def list_backend_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in titles and content"),
    limit: int = Query(50, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination")
):
    """List all documents in persistent backend storage"""
    try:
        documents = list(persistent_documents.values())
        
        # Apply filters
        if category:
            documents = [d for d in documents if d.category.lower() == category.lower()]
        
        if status:
            documents = [d for d in documents if d.status.lower() == status.lower()]
        
        if search:
            search_lower = search.lower()
            filtered_docs = []
            for doc in documents:
                # Search in title
                if search_lower in doc.title.lower():
                    filtered_docs.append(doc)
                    continue
                
                # Search in content
                content = document_contents.get(doc.id, "")
                if search_lower in content.lower():
                    filtered_docs.append(doc)
            
            documents = filtered_docs
        
        # Sort by updated_at descending
        documents.sort(key=lambda x: x.updated_at, reverse=True)
        
        # Apply pagination
        total = len(documents)
        documents = documents[offset:offset + limit]
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'documents': [doc.dict() for doc in documents],
                'total': total,
                'offset': offset,
                'limit': limit,
                'storage_type': 'backend_persistent'
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing backend documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list backend documents")

@router.post("/research/generate")
async def generate_document_from_research(
    research_query: str = Field(..., description="Research query to generate document from"),
    document_title: Optional[str] = Field(None, description="Title for generated document"),
    session_id: str = Depends(get_session_id)
):
    """Generate document using research agent (integrates with existing research system)"""
    try:
        # This would integrate with the existing research agent
        # For now, simulate the research process
        
        research_session_id = str(uuid.uuid4())
        
        # Simulate research content generation
        generated_content = f"""# Research Report: {research_query}

## Executive Summary
This document was generated by the PyGent Factory research agent based on the query: "{research_query}"

## Research Findings
[This would contain actual research findings from the research agent]

## Methodology
- Utilized PyGent Factory's research pipeline
- Integrated with Document Creation Agent
- Applied A2A protocol for agent coordination

## Generated on: {datetime.now().isoformat()}
## Research Session ID: {research_session_id}
"""
        
        # Create document using existing create endpoint
        doc_title = document_title or f"Research Report: {research_query}"
        
        create_request = DocumentCreateRequest(
            title=doc_title,
            content=generated_content,
            category="Research Generated",
            tags=["research", "ai-generated", "report"],
            research_session_id=research_session_id
        )
        
        # Create the document
        doc_id = str(uuid.uuid4())
        
        # Save to persistent backend
        file_path = f"research_documents/{doc_id}.md"
        document_contents[doc_id] = generated_content
        
        metadata = DocumentMetadata(
            id=doc_id,
            title=doc_title,
            category="Research Generated",
            file_path=file_path,
            size_bytes=len(generated_content.encode('utf-8')),
            tags=["research", "ai-generated", "report"],
            research_session_id=research_session_id,
            is_generated=True
        )
        persistent_documents[doc_id] = metadata
        
        # Create session reference
        preview_content = generated_content[:200] + "..."
        backend_link = f"/api/v1/documentation/documents/{doc_id}"
        
        session_doc = SessionDocument(
            session_id=session_id,
            document_id=doc_id,
            title=doc_title,
            preview_content=preview_content,
            backend_link=backend_link
        )
        
        if session_id not in session_documents:
            session_documents[session_id] = []
        session_documents[session_id].append(session_doc)
        
        logger.info(f"Generated research document {doc_id} for query: {research_query}")
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'document_id': doc_id,
                'research_session_id': research_session_id,
                'backend_link': backend_link,
                'session_reference': session_doc.dict(),
                'metadata': metadata.dict(),
                'generation_info': {
                    'query': research_query,
                    'generated_at': datetime.now().isoformat(),
                    'content_length': len(generated_content)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating research document: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate research document")

@router.get("/stats")
async def get_documentation_statistics():
    """Get comprehensive documentation statistics"""
    try:
        clean_expired_sessions()
        
        # Backend statistics
        backend_docs = list(persistent_documents.values())
        total_backend = len(backend_docs)
        
        categories = {}
        generated_count = 0
        total_size = 0
        
        for doc in backend_docs:
            categories[doc.category] = categories.get(doc.category, 0) + 1
            if doc.is_generated:
                generated_count += 1
            total_size += doc.size_bytes
        
        # Session statistics
        total_sessions = len(session_documents)
        total_session_docs = sum(len(docs) for docs in session_documents.values())
        
        return JSONResponse(content={
            'status': 'success',
            'data': {
                'backend_storage': {
                    'total_documents': total_backend,
                    'total_size_bytes': total_size,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'generated_documents': generated_count,
                    'manual_documents': total_backend - generated_count,
                    'categories': categories
                },
                'session_storage': {
                    'active_sessions': total_sessions,
                    'total_session_documents': total_session_docs,
                    'storage_type': 'temporary_links_only',
                    'note': 'Session documents cleared on logout'
                },
                'system_info': {
                    'storage_architecture': 'persistent_backend_with_session_links',
                    'data_retention': 'backend_permanent_session_temporary',
                    'last_updated': datetime.now().isoformat()
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting documentation statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")
