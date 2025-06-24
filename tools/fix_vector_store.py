#!/usr/bin/env python3
"""
Script to fix the missing get_collection_stats method in vector stores
"""

import os

# Method to add to FAISS vector store
faiss_method = '''
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            if collection not in self.collections:
                return {
                    "error": f"Collection '{collection}' not found",
                    "exists": False
                }
            
            # Get basic collection info
            collection_info = self.collections[collection]
            doc_storage = self.documents.get(collection, {})
            index = self.indexes.get(collection)
            
            # Calculate statistics
            document_count = len(doc_storage)
            embedding_dimension = collection_info.get("dimension", 0)
            
            # Get index statistics if available
            index_stats = {}
            if index:
                index_stats = {
                    "index_type": collection_info.get("index_type", "unknown"),
                    "is_trained": getattr(index, 'is_trained', True),
                    "ntotal": getattr(index, 'ntotal', 0)
                }
            
            # Get document metadata statistics
            metadata_fields = set()
            document_types = set()
            sources = set()
            
            for doc in doc_storage.values():
                if doc.metadata:
                    metadata_fields.update(doc.metadata.keys())
                if doc.document_type:
                    document_types.add(doc.document_type)
                if doc.source:
                    sources.add(doc.source)
            
            return {
                "exists": True,
                "document_count": document_count,
                "embedding_dimension": embedding_dimension,
                "distance_metric": collection_info.get("distance_metric", "cosine"),
                "created_at": collection_info.get("created_at"),
                "last_updated": datetime.utcnow().isoformat(),
                "index_stats": index_stats,
                "metadata_fields": list(metadata_fields),
                "document_types": list(document_types),
                "sources": list(sources),
                "collection_config": {
                    "index_type": collection_info.get("index_type", self.index_type),
                    "nlist": collection_info.get("nlist", self.nlist),
                    "nprobe": collection_info.get("nprobe", self.nprobe)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats for FAISS collection '{collection}': {str(e)}")
            return {
                "error": str(e),
                "exists": False
            }
'''

def fix_faiss_vector_store():
    """Add the missing method to FAISS vector store"""
    file_path = "src/storage/vector/faiss.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add the method before the last line if it doesn't exist
    if "get_collection_stats" not in content:
        # Find the last method end and add our method
        lines = content.split('\n')
        
        # Find the end of the last method (look for the end of count_documents)
        for i in reversed(range(len(lines))):
            if lines[i].strip() == 'return 0' and i > 500:  # Approximate location
                # Insert the method after this line
                lines.insert(i + 1, faiss_method)
                break
        
        # Write back
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Added get_collection_stats method to {file_path}")
    else:
        print(f"get_collection_stats already exists in {file_path}")

if __name__ == "__main__":
    fix_faiss_vector_store()
