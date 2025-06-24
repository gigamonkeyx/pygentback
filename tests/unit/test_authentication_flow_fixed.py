#!/usr/bin/env python3
"""
Test script to create a test user and verify authentication flow
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Test user creation and authentication
async def test_authentication():
    from src.database.connection import initialize_database
    from src.services.user_service import UserService
    from src.security.auth import UserRole
    from src.config.settings import get_settings
    
    print("Testing authentication flow...")
    
    try:
        # Initialize database
        settings = get_settings()
        db_manager = await initialize_database(settings)
        
        # Initialize user service with database manager
        user_service = UserService(db_manager)
        
        # Create test user
        print("Creating test user...")
        test_user = await user_service.create_user(
            username="testuser",
            email="testuser@example.com",
            password="testpass123",
            role=UserRole.USER
        )
        
        if test_user:
            print(f"Created user: {test_user.username} (ID: {test_user.id})")
        else:
            print("User creation failed!")
            return
        
        # Test authentication
        print("Testing authentication...")
        auth_user = await user_service.authenticate_user("testuser", "testpass123")
        if auth_user:
            print(f"Authentication successful: {auth_user.username}")
        else:
            print("Authentication failed!")
            return
        
        # Test token creation
        print("Testing JWT token creation...")
        from src.security.auth import TokenManager
        
        token_manager = TokenManager(settings)
        token = token_manager.create_access_token(auth_user)
        print(f"Generated token: {token[:50]}...")
        
        # Test token verification
        token_data = token_manager.verify_token(token)
        print(f"Token verified for user: {token_data.username}")
        
        print("✅ Authentication flow test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during authentication test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_authentication())
