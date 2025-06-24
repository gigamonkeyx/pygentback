#!/usr/bin/env python3
"""
Fix JSONB References Script

Replace all remaining JSONB references with JSONType in database models.
"""

import re

def fix_jsonb_references():
    """Fix all JSONB references in models.py"""
    
    # Read the file
    with open('src/database/models.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all JSONB references with JSONType
    # This regex finds Column(JSONB and replaces with Column(JSONType
    content = re.sub(r'Column\(JSONB([,)])', r'Column(JSONType\1', content)
    
    # Also handle cases where JSONB is used in other contexts
    content = re.sub(r'= Column\(Vector\(1536\) if VECTOR_SUPPORT else JSONB\)', 
                     r'= Column(Vector(1536) if VECTOR_SUPPORT else JSONType)', content)
    
    # Write the file back
    with open('src/database/models.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all JSONB references in database models")

if __name__ == "__main__":
    fix_jsonb_references()
