#!/usr/bin/env python3
"""
Validate Production Database Models

Validate the production database models structure and relationships.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database.models import (
    Base, User, Agent, Task, Document, AgentType, TaskState, 
    TaskArtifact, AgentMemory, DocumentChunk, DocumentEmbedding,
    CodeRepository, CodeFile, CodeEmbedding, UserSession, 
    SystemConfiguration, PRODUCTION_MODEL_REGISTRY
)

def validate_model_structure():
    """Validate model structure and attributes"""
    print("🔍 Validating Production Model Structure...")
    
    issues = []
    
    # Check all models are in registry
    expected_models = [
        User, Agent, Task, TaskArtifact, AgentMemory, Document, 
        DocumentChunk, DocumentEmbedding, CodeRepository, CodeFile, 
        CodeEmbedding, UserSession, SystemConfiguration
    ]
    
    for model in expected_models:
        model_name = model.__name__
        if model_name not in PRODUCTION_MODEL_REGISTRY:
            issues.append(f"Model {model_name} not in registry")
        else:
            print(f"✅ {model_name} registered")
    
    # Check model attributes
    required_attributes = {
        'User': ['id', 'username', 'email', 'created_at', 'updated_at'],
        'Agent': ['id', 'name', 'agent_type', 'capabilities', 'created_at'],
        'Task': ['id', 'task_type', 'state', 'input_data', 'output_data'],
        'Document': ['id', 'title', 'content', 'content_type', 'processing_status'],
        'CodeRepository': ['id', 'name', 'repository_url', 'analysis_status'],
        'CodeFile': ['id', 'file_path', 'filename', 'language', 'content']
    }
    
    for model_name, attrs in required_attributes.items():
        model = PRODUCTION_MODEL_REGISTRY.get(model_name)
        if model:
            for attr in attrs:
                if not hasattr(model, attr):
                    issues.append(f"Model {model_name} missing attribute {attr}")
                else:
                    print(f"✅ {model_name}.{attr} exists")
    
    # Check relationships
    relationship_checks = [
        (User, 'agents', 'User should have agents relationship'),
        (Agent, 'tasks', 'Agent should have tasks relationship'),
        (Task, 'artifacts', 'Task should have artifacts relationship'),
        (Document, 'embeddings', 'Document should have embeddings relationship'),
        (CodeRepository, 'code_files', 'CodeRepository should have code_files relationship')
    ]
    
    for model, rel_name, description in relationship_checks:
        if hasattr(model, rel_name):
            print(f"✅ {description}")
        else:
            issues.append(description)
    
    return issues


def validate_enums():
    """Validate enum definitions"""
    print("\n🔢 Validating Enums...")
    
    issues = []
    
    # Check AgentType enum
    expected_agent_types = ['ORCHESTRATOR', 'DOCUMENT_PROCESSOR', 'VECTOR_SEARCH', 'A2A_AGENT', 'CODE_ANALYZER', 'CODE_GENERATOR']
    for agent_type in expected_agent_types:
        if hasattr(AgentType, agent_type):
            print(f"✅ AgentType.{agent_type} exists")
        else:
            issues.append(f"AgentType.{agent_type} missing")
    
    # Check TaskState enum
    expected_task_states = ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED']
    for task_state in expected_task_states:
        if hasattr(TaskState, task_state):
            print(f"✅ TaskState.{task_state} exists")
        else:
            issues.append(f"TaskState.{task_state} missing")
    
    return issues


def validate_indexes():
    """Validate database indexes"""
    print("\n📊 Validating Database Indexes...")
    
    issues = []
    
    # Check that models have __table_args__ with indexes
    models_with_indexes = [User, Agent, Task, Document, CodeRepository, CodeFile]
    
    for model in models_with_indexes:
        if hasattr(model, '__table_args__'):
            table_args = model.__table_args__
            if table_args and len(table_args) > 0:
                print(f"✅ {model.__name__} has indexes defined")
            else:
                issues.append(f"{model.__name__} missing indexes")
        else:
            issues.append(f"{model.__name__} missing __table_args__")
    
    return issues


def validate_metadata_columns():
    """Validate that metadata columns don't conflict with SQLAlchemy"""
    print("\n🏷️ Validating Metadata Columns...")
    
    issues = []
    
    # Check that no model uses 'metadata' as column name (conflicts with SQLAlchemy)
    for model_name, model in PRODUCTION_MODEL_REGISTRY.items():
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'type'):
            # This would be a SQLAlchemy Column, not the class metadata
            issues.append(f"{model_name} has conflicting 'metadata' column")
        else:
            print(f"✅ {model_name} has no metadata column conflicts")
    
    # Check for proper metadata column naming
    metadata_columns = {
        'User': 'user_metadata',
        'Task': 'task_metadata', 
        'Document': 'document_metadata',
        'Agent': None,  # Agent doesn't need metadata column
        'TaskArtifact': 'artifact_metadata'
    }
    
    for model_name, expected_col in metadata_columns.items():
        if expected_col:
            model = PRODUCTION_MODEL_REGISTRY.get(model_name)
            if model and hasattr(model, expected_col):
                print(f"✅ {model_name} has {expected_col} column")
            else:
                issues.append(f"{model_name} missing {expected_col} column")
    
    return issues


def validate_foreign_keys():
    """Validate foreign key relationships"""
    print("\n🔗 Validating Foreign Keys...")
    
    issues = []
    
    # Check foreign key definitions
    fk_checks = [
        (Agent, 'user_id', 'Agent should reference User'),
        (Task, 'agent_id', 'Task should reference Agent'),
        (TaskArtifact, 'task_id', 'TaskArtifact should reference Task'),
        (AgentMemory, 'agent_id', 'AgentMemory should reference Agent'),
        (Document, 'user_id', 'Document should reference User'),
        (DocumentChunk, 'document_id', 'DocumentChunk should reference Document'),
        (DocumentEmbedding, 'document_id', 'DocumentEmbedding should reference Document'),
        (CodeFile, 'repository_id', 'CodeFile should reference CodeRepository'),
        (CodeEmbedding, 'code_file_id', 'CodeEmbedding should reference CodeFile'),
        (UserSession, 'user_id', 'UserSession should reference User')
    ]
    
    for model, fk_column, description in fk_checks:
        if hasattr(model, fk_column):
            print(f"✅ {description}")
        else:
            issues.append(f"{description} - missing {fk_column}")
    
    return issues


def validate_production_readiness():
    """Validate production readiness features"""
    print("\n🏭 Validating Production Readiness...")
    
    issues = []
    
    # Check for UUID primary keys (production best practice)
    uuid_models = [User, Agent, Task, Document, CodeRepository]
    for model in uuid_models:
        if hasattr(model, 'id'):
            # Check if it's likely a UUID column (can't easily check type without DB)
            print(f"✅ {model.__name__} has id column (assuming UUID)")
        else:
            issues.append(f"{model.__name__} missing id column")
    
    # Check for timestamp columns
    timestamp_models = [User, Agent, Task, Document, CodeRepository, CodeFile]
    for model in timestamp_models:
        has_created = hasattr(model, 'created_at')
        has_updated = hasattr(model, 'updated_at')
        
        if has_created and has_updated:
            print(f"✅ {model.__name__} has timestamp columns")
        else:
            missing = []
            if not has_created:
                missing.append('created_at')
            if not has_updated:
                missing.append('updated_at')
            issues.append(f"{model.__name__} missing timestamp columns: {missing}")
    
    return issues


def main():
    """Run all validation tests"""
    print("🚀 Production Database Models Validation")
    print("=" * 60)
    
    all_issues = []
    
    # Run all validation tests
    validation_tests = [
        ("Model Structure", validate_model_structure),
        ("Enums", validate_enums),
        ("Database Indexes", validate_indexes),
        ("Metadata Columns", validate_metadata_columns),
        ("Foreign Keys", validate_foreign_keys),
        ("Production Readiness", validate_production_readiness)
    ]
    
    for test_name, test_func in validation_tests:
        print(f"\n{test_name}:")
        issues = test_func()
        all_issues.extend(issues)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("   Production database models are properly structured.")
        print("   Ready for PostgreSQL deployment.")
        return True
    else:
        print(f"⚠️ FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        print("\n   Fix these issues before production deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
