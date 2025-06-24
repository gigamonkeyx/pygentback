"""
Document Service - Database-backed document management

This module provides database-backed document management services, ensuring
all documents are associated with users and research sessions are tracked.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database.models import (
    Document as DocumentModel, 
    DocumentationFile, 
    DocumentationVersion, 
    DocumentationTag,
    User
)
from ..database.connection import get_database_session


logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents with database persistence and user association."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize DocumentService with optional database session."""
        self._db_session = db_session
    
    def _get_session(self) -> Session:
        """Get database session - use injected session or create new one."""
        if self._db_session:
            return self._db_session
        return next(get_database_session())
    
    def _should_close_session(self) -> bool:
        """Check if we should close the session (only if we created it)."""
        return self._db_session is None
    
    def create_document(
        self,
        user_id: str,
        title: str,
        content: str,
        document_type: str,
        source_url: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        research_session_id: Optional[str] = None
    ) -> DocumentModel:
        """
        Create a new document associated with a user.
        
        Args:
            user_id: ID of the user creating the document
            title: Document title
            content: Document content
            document_type: Type of document (e.g., 'markdown', 'text', 'research')
            source_url: Optional source URL
            meta_data: Optional metadata dictionary
            research_session_id: Optional research session ID if created by research agent
            
        Returns:
            DocumentModel: The created document database record
            
        Raises:
            ValueError: If user not found
            SQLAlchemyError: If database operation fails
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            # Verify user exists
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User not found: {user_id}")
            
            # Add research session info to metadata if provided
            if meta_data is None:
                meta_data = {}
            
            if research_session_id:
                meta_data['research_session_id'] = research_session_id
            
            # Create document record
            document = DocumentModel(
                user_id=user_id,
                title=title,
                content=content,
                document_type=document_type,
                source_url=source_url,
                meta_data=meta_data
            )
            
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"Created document {document.id} for user {user_id}")
            return document
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error creating document: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating document: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def create_documentation_file(
        self,
        user_id: str,
        title: str,
        content: str,
        file_path: str,
        category: str,
        research_session_id: Optional[str] = None,
        agent_workflow_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DocumentationFile:
        """
        Create a new documentation file associated with a user.
        
        Args:
            user_id: ID of the user creating the documentation
            title: Documentation title
            content: Documentation content (markdown)
            file_path: File path for the documentation
            category: Documentation category
            research_session_id: Optional research session ID
            agent_workflow_id: Optional agent workflow ID
            tags: Optional list of tags
            
        Returns:
            DocumentationFile: The created documentation file record
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            # Verify user exists
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User not found: {user_id}")
            
            # Build metadata
            meta_data = {}
            if research_session_id:
                meta_data['research_session_id'] = research_session_id
            if agent_workflow_id:
                meta_data['agent_workflow_id'] = agent_workflow_id
                meta_data['generated_by_agent'] = True
            
            # Create documentation file record
            doc_file = DocumentationFile(
                user_id=user_id,
                title=title,
                content=content,
                file_path=file_path,
                category=category,
                meta_data=meta_data
            )
            
            session.add(doc_file)
            session.flush()  # Get the ID for tags
            
            # Add tags if provided
            if tags:
                for tag_name in tags:
                    tag = DocumentationTag(
                        documentation_file_id=doc_file.id,
                        tag_name=tag_name.strip()
                    )
                    session.add(tag)
            
            # Create initial version
            version = DocumentationVersion(
                documentation_file_id=doc_file.id,
                version_number=1,
                content=content,
                change_summary="Initial creation",
                created_by=user_id
            )
            session.add(version)
            
            session.commit()
            session.refresh(doc_file)
            
            logger.info(f"Created documentation file {doc_file.id} for user {user_id}")
            return doc_file
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error creating documentation file: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating documentation file: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def update_documentation_file(
        self,
        user_id: str,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        change_summary: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[DocumentationFile]:
        """
        Update a documentation file with version tracking.
        
        Args:
            user_id: ID of the user (for authorization)
            document_id: ID of the documentation file
            title: New title (if provided)
            content: New content (if provided)
            category: New category (if provided)
            change_summary: Summary of changes
            tags: New tags list (if provided)
            
        Returns:
            DocumentationFile or None: Updated documentation file if found and owned by user
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            doc_file = session.query(DocumentationFile).filter(
                DocumentationFile.id == document_id,
                DocumentationFile.user_id == user_id
            ).first()
            
            if not doc_file:
                return None
            
            # Check if content is actually changing
            content_changed = content is not None and content != doc_file.content
            
            # Update fields
            if title is not None:
                doc_file.title = title
            
            if content is not None:
                doc_file.content = content
            
            if category is not None:
                doc_file.category = category
            
            doc_file.updated_at = datetime.utcnow()
            
            # Update tags if provided
            if tags is not None:
                # Remove existing tags
                session.query(DocumentationTag).filter(
                    DocumentationTag.documentation_file_id == document_id
                ).delete()
                
                # Add new tags
                for tag_name in tags:
                    tag = DocumentationTag(
                        documentation_file_id=document_id,
                        tag_name=tag_name.strip()
                    )
                    session.add(tag)
            
            # Create new version if content changed
            if content_changed:
                # Get the next version number
                latest_version = session.query(DocumentationVersion).filter(
                    DocumentationVersion.documentation_file_id == document_id
                ).order_by(DocumentationVersion.version_number.desc()).first()
                
                next_version = (latest_version.version_number + 1) if latest_version else 1
                
                version = DocumentationVersion(
                    documentation_file_id=document_id,
                    version_number=next_version,
                    content=content,
                    change_summary=change_summary or "Content updated",
                    created_by=user_id
                )
                session.add(version)
                
                # Update version number in metadata
                if not doc_file.meta_data:
                    doc_file.meta_data = {}
                doc_file.meta_data['version_number'] = next_version
            
            session.commit()
            session.refresh(doc_file)
            
            logger.info(f"Updated documentation file {document_id} for user {user_id}")
            return doc_file
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error updating documentation file: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating documentation file: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def get_documentation_file_by_id(self, document_id: str, user_id: str) -> Optional[DocumentationFile]:
        """
        Get a documentation file by ID, ensuring it belongs to the specified user.
        
        Args:
            document_id: ID of the documentation file
            user_id: ID of the user (for authorization)
            
        Returns:
            DocumentationFile or None: The documentation file if found and owned by user
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            doc_file = session.query(DocumentationFile).filter(
                DocumentationFile.id == document_id,
                DocumentationFile.user_id == user_id
            ).first()
            return doc_file
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documentation file: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def list_documentation_files(
        self,
        user_id: str,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get documentation files for a user with filtering and pagination.
        
        Args:
            user_id: ID of the user
            category: Optional category filter
            search_query: Optional search query (searches title and content)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dict containing documents list and total count
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            query = session.query(DocumentationFile).filter(DocumentationFile.user_id == user_id)
            
            if category:
                query = query.filter(DocumentationFile.category == category)
            
            if search_query:
                search_term = f"%{search_query}%"
                query = query.filter(
                    DocumentationFile.title.ilike(search_term) |
                    DocumentationFile.content.ilike(search_term)
                )
            
            # Get total count
            total = query.count()
            
            # Apply pagination and ordering
            documents = query.order_by(DocumentationFile.updated_at.desc()).offset(offset).limit(limit).all()
            
            return {
                'documents': documents,
                'total': total,
                'limit': limit,
                'offset': offset
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error listing documentation files: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()

    def get_user_documents(
        self,
        user_id: str,
        document_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[DocumentModel]:
        """
        Get all documents for a specific user with optional filtering.
        
        Args:
            user_id: ID of the user
            document_type: Optional document type filter
            limit: Optional limit on number of results
            
        Returns:
            List[DocumentModel]: List of user's documents
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            query = session.query(DocumentModel).filter(DocumentModel.user_id == user_id)
            
            if document_type:
                query = query.filter(DocumentModel.document_type == document_type)
            
            query = query.order_by(DocumentModel.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            documents = query.all()
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user documents: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def get_document_by_id(self, document_id: str, user_id: str) -> Optional[DocumentModel]:
        """
        Get a document by ID, ensuring it belongs to the specified user.
        
        Args:
            document_id: ID of the document
            user_id: ID of the user (for authorization)
            
        Returns:
            DocumentModel or None: The document if found and owned by user
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            document = session.query(DocumentModel).filter(
                DocumentModel.id == document_id,
                DocumentModel.user_id == user_id
            ).first()
            return document
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting document: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def update_document(
        self,
        document_id: str,
        user_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Optional[DocumentModel]:
        """
        Update a document.
        
        Args:
            document_id: ID of the document
            user_id: ID of the user (for authorization)
            title: New title (if provided)
            content: New content (if provided)
            meta_data: New metadata (if provided)
            
        Returns:
            DocumentModel or None: Updated document if found and owned by user
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            document = session.query(DocumentModel).filter(
                DocumentModel.id == document_id,
                DocumentModel.user_id == user_id
            ).first()
            
            if not document:
                return None
            
            if title is not None:
                document.title = title
            
            if content is not None:
                document.content = content
            
            if meta_data is not None:
                # Merge metadata instead of replacing
                current_meta = document.meta_data or {}
                current_meta.update(meta_data)
                document.meta_data = current_meta
            
            document.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(document)
            
            logger.info(f"Updated document {document_id}")
            return document
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error updating document: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: ID of the document
            user_id: ID of the user (for authorization)
            
        Returns:
            bool: True if document was deleted, False if not found
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            document = session.query(DocumentModel).filter(
                DocumentModel.id == document_id,
                DocumentModel.user_id == user_id
            ).first()
            
            if not document:
                return False
            
            session.delete(document)
            session.commit()
            
            logger.info(f"Deleted document {document_id} for user {user_id}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error deleting document: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def search_user_documents(
        self,
        user_id: str,
        query: str,
        document_type: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> List[DocumentModel]:
        """
        Search documents for a user by title and content.
        
        Args:
            user_id: ID of the user
            query: Search query
            document_type: Optional document type filter
            limit: Maximum number of results
            
        Returns:
            List[DocumentModel]: Matching documents
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            # Simple text search (can be enhanced with full-text search later)
            search_filter = session.query(DocumentModel).filter(
                DocumentModel.user_id == user_id
            ).filter(
                (DocumentModel.title.ilike(f"%{query}%")) |
                (DocumentModel.content.ilike(f"%{query}%"))
            )
            
            if document_type:
                search_filter = search_filter.filter(DocumentModel.document_type == document_type)
            
            documents = search_filter.order_by(
                DocumentModel.updated_at.desc()
            ).limit(limit).all()
            
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error searching documents: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def get_research_session_documents(
        self,
        user_id: str,
        research_session_id: str
    ) -> List[DocumentModel]:
        """
        Get all documents associated with a specific research session.
        
        Args:
            user_id: ID of the user
            research_session_id: ID of the research session
            
        Returns:
            List[DocumentModel]: Documents from the research session
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            documents = session.query(DocumentModel).filter(
                DocumentModel.user_id == user_id,
                DocumentModel.meta_data.contains({"research_session_id": research_session_id})
            ).order_by(DocumentModel.created_at.asc()).all()
            
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting research session documents: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()
    
    def get_document_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's documents.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dict with document statistics
        """
        session = self._get_session()
        should_close = self._should_close_session()
        
        try:
            documents = session.query(DocumentModel).filter(DocumentModel.user_id == user_id).all()
            
            stats = {
                'total_documents': len(documents),
                'documents_by_type': {},
                'recent_documents': [],
                'total_content_length': 0
            }
            
            for doc in documents:
                # Count by type
                stats['documents_by_type'][doc.document_type] = \
                    stats['documents_by_type'].get(doc.document_type, 0) + 1
                
                # Total content length
                stats['total_content_length'] += len(doc.content or '')
            
            # Get recent documents (last 10)
            recent_docs = sorted(documents, key=lambda d: d.updated_at, reverse=True)[:10]
            stats['recent_documents'] = [
                {
                    'id': doc.id,
                    'title': doc.title,
                    'type': doc.document_type,
                    'updated_at': doc.updated_at.isoformat()
                }
                for doc in recent_docs
            ]
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting document statistics: {str(e)}")
            raise
        finally:
            if should_close:
                session.close()


# Global document service instance
_document_service: Optional[DocumentService] = None


def set_document_service(service: DocumentService):
    """Set the global document service instance"""
    global _document_service
    _document_service = service


def get_document_service() -> DocumentService:
    """Get the document service dependency"""
    if _document_service is None:
        raise RuntimeError("Document service not initialized")
    return _document_service
