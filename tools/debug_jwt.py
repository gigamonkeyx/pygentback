#!/usr/bin/env python3
"""
Debug JWT Issue
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def debug_jwt():
    """Debug JWT validation issue"""
    try:
        from a2a_protocol.security import A2ASecurityManager
        import time
        import jwt
        import datetime
        
        print("🔍 Debugging JWT Validation Issue")
        print("=" * 50)
        
        security = A2ASecurityManager()
        payload = {'user_id': 'test_user', 'scope': 'agent:read'}
        
        print("1. Creating token...")
        token = security.generate_jwt_token(payload)
        print(f"   Token created: {token[:50]}...")
        
        print("\n2. Decoding token without verification...")
        decoded = jwt.decode(token, options={'verify_signature': False})
        print(f"   Token payload: {decoded}")
        
        print("\n3. Checking timestamps...")
        now = datetime.datetime.utcnow()
        current_timestamp = int(now.timestamp())
        token_iat = decoded.get('iat')
        token_exp = decoded.get('exp')
        
        print(f"   Current time: {current_timestamp}")
        print(f"   Token iat: {token_iat}")
        print(f"   Token exp: {token_exp}")
        print(f"   Time diff (current - iat): {current_timestamp - token_iat}")
        
        print("\n4. Testing validation immediately...")
        result1 = security.validate_jwt_token(token)
        print(f"   Immediate validation: success={result1.success}")
        if not result1.success:
            print(f"   Error: {result1.error}")
        
        print("\n5. Testing validation after 1 second...")
        time.sleep(1)
        result2 = security.validate_jwt_token(token)
        print(f"   After 1 second: success={result2.success}")
        if not result2.success:
            print(f"   Error: {result2.error}")
        
        print("\n6. Testing with manual JWT creation...")
        # Create a token manually with proper timing
        manual_now = datetime.datetime.utcnow()
        manual_payload = {
            "sub": "test_user",
            "iss": security.jwt_config.issuer,
            "aud": security.jwt_config.audience,
            "iat": int(manual_now.timestamp()) - 5,  # 5 seconds ago
            "exp": int((manual_now + datetime.timedelta(minutes=30)).timestamp()),
            "jti": "test-jti",
            "scopes": ["agent:read"],
            "type": "access"
        }
        
        manual_token = jwt.encode(
            manual_payload, 
            security.jwt_config.secret_key, 
            algorithm=security.jwt_config.algorithm
        )
        
        print(f"   Manual token created: {manual_token[:50]}...")
        
        result3 = security.validate_jwt_token(manual_token)
        print(f"   Manual token validation: success={result3.success}")
        if not result3.success:
            print(f"   Error: {result3.error}")
        
        print("\n" + "=" * 50)
        if result3.success:
            print("✅ JWT validation works with proper timing!")
            return True
        else:
            print("❌ JWT validation still failing")
            return False
            
    except Exception as e:
        print(f"❌ Error during JWT debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_jwt()
    sys.exit(0 if success else 1)
