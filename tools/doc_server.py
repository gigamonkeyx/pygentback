"""
Minimal FastAPI server for testing documentation endpoints
"""
import sys
sys.path.insert(0, 'src')

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional
import uvicorn

# Import our modules
from src.database.models import User
from src.database.connection import get_sync_db_session, init_sync_database
from src.services.document_service import DocumentService
from src.api.routes.documentation import DocumentCreationRequest, DocumentUpdateRequest

app = FastAPI(title="PyGent Documentation API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_test_user(db: Session = Depends(get_sync_db_session)) -> User:
    """Get or create test user for demonstration"""
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
    """Initialize database"""
    init_sync_database()
    print("✅ Database initialized")

@app.get("/")
async def root():
    return {"message": "PyGent Documentation API is running", "status": "healthy"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "documentation-api"}

# Documentation endpoints
@app.get("/api/documentation/persistent")
async def list_documentation_files(
    skip: int = 0,
    limit: int = 50,
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    filters = {}
    if category:
        filters["category"] = category
    if file_type:
        filters["file_type"] = file_type
    
    result = doc_service.list_documentation_files(
        current_user.id, 
        skip=skip, 
        limit=limit, 
        filters=filters
    )
    
    # Format response for frontend compatibility
    return {
        "status": "success",
        "data": {
            "documents": result.get("files", [])
        }
    }

@app.get("/api/documentation/persistent/{doc_id}")
async def get_documentation_file(
    doc_id: str,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    doc = doc_service.get_documentation_file(doc_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    
    # Format response for frontend compatibility
    return {
        "status": "success",
        "data": doc
    }

@app.post("/api/documentation/persistent")
async def create_documentation_file(
    request: DocumentCreationRequest,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    doc_data = {
        "title": request.title,
        "content": request.content,
        "file_type": request.file_type or "markdown",
        "tags": request.tags or [],
        "category": request.category or "general",
        "is_public": request.is_public if request.is_public is not None else False,
        "agent_id": request.agent_id,
        "research_session_id": request.research_session_id
    }
    
    result = doc_service.create_documentation_file(current_user.id, doc_data)
    return {
        "status": "success",
        "data": result
    }

@app.get("/api/documentation/categories")
async def get_documentation_categories(
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    result = doc_service.list_documentation_files(current_user.id)
    
    # Extract categories from files
    categories = {}
    for file in result.get("files", []):
        category = file.get("category", "uncategorized")
        categories[category] = categories.get(category, 0) + 1
    
    category_list = [{"name": name, "count": count} for name, count in categories.items()]
    
    return {
        "status": "success",
        "data": {
            "categories": category_list
        }
    }

@app.get("/api/documentation/search")
async def search_documentation(
    query: str,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    # For now, do a simple search by title/content
    result = doc_service.list_documentation_files(current_user.id)
    
    # Filter files that match the query
    matching_files = []
    for file in result.get("files", []):
        if (query.lower() in file.get("title", "").lower() or 
            query.lower() in file.get("content", "").lower()):
            matching_files.append(file)
    
    return {
        "status": "success",
        "data": {
            "documents": matching_files
        }
    }

# Legacy endpoints for backward compatibility
@app.post("/api/documentation/files")
async def create_documentation_file_legacy(
    request: DocumentCreationRequest,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    return await create_documentation_file(request, db, current_user)

@app.get("/api/documentation/files/{doc_id}")
async def get_documentation_file_legacy(
    doc_id: str,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    doc = doc_service.get_documentation_file(doc_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return doc

@app.get("/api/documentation/files")
async def list_documentation_files_legacy(
    skip: int = 0,
    limit: int = 50,
    category: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    filters = {}
    if category:
        filters["category"] = category
    if file_type:
        filters["file_type"] = file_type
    
    result = doc_service.list_documentation_files(
        current_user.id, 
        skip=skip, 
        limit=limit, 
        filters=filters
    )
    return result

@app.put("/api/documentation/files/{doc_id}")
async def update_documentation_file(
    doc_id: str,
    request: DocumentUpdateRequest,
    db: Session = Depends(get_sync_db_session),
    current_user: User = Depends(get_test_user)
):
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
    current_user: User = Depends(get_test_user)
):
    doc_service = DocumentService(db)
    success = doc_service.delete_documentation_file(doc_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return {"message": "Documentation file deleted successfully"}

if __name__ == "__main__":
    print("🚀 Starting PyGent Documentation API server...")
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
