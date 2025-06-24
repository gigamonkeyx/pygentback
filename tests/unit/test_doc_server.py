"""
Test server for documentation endpoints
"""
import sys
sys.path.insert(0, 'src')

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import Optional
import uvicorn

# Import database models and services
from src.database.models import User
from src.database.connection import get_sync_db_session, init_sync_database
from src.services.document_service import DocumentService
from src.api.routes.documentation import CreateDocumentationFileRequest, UpdateDocumentationFileRequest

app = FastAPI(title="PyGent Documentation API Test", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple auth
security = HTTPBearer()

def get_current_user(db: Session = Depends(get_sync_db_session)) -> User:
    """Get or create a test user"""
    user = db.query(User).filter(User.email == "test@example.com").first()
    if not user:
        user = User(
            email="test@example.com",
            username="testuser", 
            full_name="Test User",
            is_active=True,
            oauth_providers=[]
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    init_sync_database()
    
@app.get("/")
async def root():
    return {"message": "Documentation API test server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Documentation endpoints
@app.post("/api/documentation/files")
async def create_documentation_file(
    request: CreateDocumentationFileRequest,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_current_user)
):
    """Create a new documentation file"""
    doc_service = DocumentService(db)
    doc_data = {
        "title": request.title,
        "content": request.content,
        "file_type": request.file_type,
        "tags": request.tags,
        "category": request.category,
        "is_public": request.is_public,
        "agent_id": request.agent_id,
        "research_session_id": request.research_session_id
    }
    return doc_service.create_documentation_file(current_user.id, doc_data)

@app.get("/api/documentation/files/{doc_id}")
async def get_documentation_file(
    doc_id: str,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get a documentation file by ID"""
    doc_service = DocumentService(db)
    doc = doc_service.get_documentation_file(doc_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return doc

@app.get("/api/documentation/files")
async def list_documentation_files(
    skip: int = 0,
    limit: int = 50,
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_current_user)
):
    """List documentation files for the current user"""
    doc_service = DocumentService(db)
    filters = {}
    if category:
        filters["category"] = category
    if file_type:
        filters["file_type"] = file_type
    
    return doc_service.list_documentation_files(
        current_user.id, 
        skip=skip, 
        limit=limit, 
        filters=filters
    )

@app.put("/api/documentation/files/{doc_id}")
async def update_documentation_file(
    doc_id: str,
    request: UpdateDocumentationFileRequest,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_current_user)
):
    """Update a documentation file"""
    doc_service = DocumentService(db)
    update_data = {}
    if request.title is not None:
        update_data["title"] = request.title
    if request.content is not None:
        update_data["content"] = request.content
    if request.tags is not None:
        update_data["tags"] = request.tags
    if request.category is not None:
        update_data["category"] = request.category
    if request.is_public is not None:
        update_data["is_public"] = request.is_public
    
    doc = doc_service.update_documentation_file(doc_id, current_user.id, update_data)
    if not doc:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return doc

@app.delete("/api/documentation/files/{doc_id}")
async def delete_documentation_file(
    doc_id: str,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_current_user)
):
    """Delete a documentation file"""
    doc_service = DocumentService(db)
    success = doc_service.delete_documentation_file(doc_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return {"message": "Documentation file deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
