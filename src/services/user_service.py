"""
User Service - Database-backed user management

Provides                # Prepare OAuth providers data
                oauth_providers_data = []
                if oauth_provider and oauth_user_id:
                    oauth_providers_data.append({
                        'provider': oauth_provider,
                        'user_id': oauth_user_id,
                        'user_info': oauth_user_info or {}
                    })
                
                # Create new user
                db_user = UserModel(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    role=role.value,
                    is_active=True,
                    oauth_providers=oauth_providers_data
                )nt creation, authentication, and management using persistent database storage.
Integrates with OAuth and supports email/password fallback authentication.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError

from ..database.models import User as UserModel
from ..security.auth import User, UserRole, PasswordManager


logger = logging.getLogger(__name__)


class UserService:
    """Database-backed user management service"""
    
    def __init__(self, database_manager=None):
        self.database_manager = database_manager
        self.password_manager = PasswordManager()
    
    async def _get_db_manager(self):
        """Get database manager, initializing if needed"""
        if self.database_manager is None:
            from ..database.connection import get_database_manager
            self.database_manager = await get_database_manager()
        return self.database_manager
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        password: Optional[str] = None,
        role: UserRole = UserRole.USER,
        oauth_provider: Optional[str] = None,
        oauth_user_id: Optional[str] = None,
        oauth_user_info: Optional[Dict[str, Any]] = None
    ) -> Optional[User]:
        """Create a new user account"""
        try:
            async with self.database_manager.get_session() as session:
                # Check if user already exists
                stmt = select(UserModel).where(
                    (UserModel.username == username) | (UserModel.email == email)
                )
                result = await session.execute(stmt)
                existing_user = result.scalar_one_or_none()
                
                if existing_user:
                    logger.warning(f"User already exists: {username} / {email}")
                    return None
                
                # Hash password if provided
                password_hash = None
                if password:
                    password_hash = self.password_manager.hash_password(password)
                
                # Create user
                db_user = UserModel(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    role=role.value,
                    is_active=True,
                    oauth_provider=oauth_provider,
                    oauth_user_id=oauth_user_id,
                    oauth_user_info=oauth_user_info or {}
                )
                
                session.add(db_user)
                await session.commit()
                await session.refresh(db_user)
                
                # Convert to Pydantic model
                return self._db_user_to_pydantic(db_user)
                
        except IntegrityError as e:
            logger.error(f"User creation failed - integrity error: {e}")
            return None
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.id == user_id)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    return self._db_user_to_pydantic(db_user)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.username == username)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    return self._db_user_to_pydantic(db_user)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.email == email)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    return self._db_user_to_pydantic(db_user)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_oauth(self, provider: str, oauth_user_id: str) -> Optional[User]:
        """Get user by OAuth provider and user ID"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(
                    and_(
                        UserModel.oauth_provider == provider,
                        UserModel.oauth_user_id == oauth_user_id
                    )
                )
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    return self._db_user_to_pydantic(db_user)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by OAuth {provider}/{oauth_user_id}: {e}")
            return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        try:
            user = await self.get_user_by_username(username)
            if not user:
                # Try email as username
                user = await self.get_user_by_email(username)
            
            if not user or not user.is_active:
                return None
            
            # Get password hash from database
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel.password_hash).where(UserModel.id == user.id)
                result = await session.execute(stmt)
                password_hash = result.scalar_one_or_none()
                
                if not password_hash:
                    return None
                
                if self.password_manager.verify_password(password, password_hash):
                    # Update last login
                    await self.update_last_login(user.id)
                    return user
                
                return None
                
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.id == user_id)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    db_user.last_login = datetime.utcnow()
                    await session.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")
            return False
    
    async def update_user(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[User]:
        """Update user fields"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.id == user_id)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if not db_user:
                    return None
                
                # Apply updates
                for field, value in updates.items():
                    if hasattr(db_user, field):
                        setattr(db_user, field, value)
                
                await session.commit()
                await session.refresh(db_user)
                
                return self._db_user_to_pydantic(db_user)
                
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return None
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user account"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel).where(UserModel.id == user_id)
                result = await session.execute(stmt)
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    await session.delete(db_user)
                    await session.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def list_users(
        self, 
        limit: int = 100, 
        offset: int = 0,
        active_only: bool = True
    ) -> List[User]:
        """List users with pagination"""
        try:
            async with self.database_manager.get_session() as session:
                stmt = select(UserModel)
                
                if active_only:
                    stmt = stmt.where(UserModel.is_active)
                
                stmt = stmt.offset(offset).limit(limit)
                
                result = await session.execute(stmt)
                db_users = result.scalars().all()
                
                return [self._db_user_to_pydantic(db_user) for db_user in db_users]
                
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    def _db_user_to_pydantic(self, db_user: UserModel) -> User:
        """Convert database user model to Pydantic model"""
        return User(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            role=UserRole(db_user.role),
            is_active=db_user.is_active,
            created_at=db_user.created_at,
            last_login=db_user.last_login,
            oauth_provider=db_user.oauth_provider,
            oauth_user_id=db_user.oauth_user_id,
            oauth_user_info=db_user.oauth_user_info or {}
        )
    
    async def create_default_users(self):
        """Create default system users if they don't exist"""
        try:
            # Create admin user
            admin_exists = await self.get_user_by_username("admin")
            if not admin_exists:
                await self.create_user(
                    username="admin",
                    email="admin@pygent.factory",
                    password="admin",  # Change in production
                    role=UserRole.ADMIN
                )
                logger.info("Created default admin user")
            
            # Create service user
            service_exists = await self.get_user_by_username("service")
            if not service_exists:
                await self.create_user(
                    username="service",
                    email="service@pygent.factory",
                    role=UserRole.SERVICE
                )
                logger.info("Created default service user")
                
        except Exception as e:
            logger.error(f"Failed to create default users: {e}")


# Global user service instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get or create the global user service instance"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
