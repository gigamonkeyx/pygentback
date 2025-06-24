import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface DocumentationFile {
  id: string;
  title: string;
  path: string;
  category: string;
  size: number;
  created_at: string;
  updated_at: string;
  generated_by_agent: boolean;
  research_session_id?: string;
  backend_stored: boolean;
  temporary_copy: boolean;
}

interface DocumentContent {
  id: string;
  title: string;
  content: string;
  html_content?: string;
  category: string;
  path: string;
  size: number;
  modified: string;
  versions?: Array<{
    version_number: number;
    created_at: string;
    change_summary: string;
    created_by: string;
  }>;
  tags?: string[];
  user_id: string;
  backend_stored: boolean;
  temporary_copy: boolean;
}

interface DocumentationCategory {
  name: string;
  count: number;
}

/**
 * DocumentationPageV2 Component
 * 
 * New documentation page that uses the backend API endpoints for persistent,
 * user-scoped documentation management with research agent integration.
 */
export const DocumentationPageV2: React.FC = () => {
  const navigate = useNavigate();
  
  const [files, setFiles] = useState<DocumentationFile[]>([]);
  const [categories, setCategories] = useState<DocumentationCategory[]>([]);
  const [selectedFile, setSelectedFile] = useState<DocumentContent | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load documentation files from backend
  const loadDocumentationFiles = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Load persistent documents from backend
      const response = await fetch('/api/documentation/persistent', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          navigate('/login');
          return;
        }
        throw new Error(`Failed to load documentation: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setFiles(data.data.documents);
      } else {
        throw new Error(data.error || 'Failed to load documentation');
      }

    } catch (err) {
      console.error('Error loading documentation:', err);
      setError(err instanceof Error ? err.message : 'Failed to load documentation');
    } finally {
      setIsLoading(false);
    }
  };

  // Load documentation categories
  const loadCategories = async () => {
    try {
      const response = await fetch('/api/documentation/categories');
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          setCategories([{ name: 'All', count: 0 }, ...data.data.categories]);
        }
      }
    } catch (err) {
      console.error('Error loading categories:', err);
    }
  };

  // Load specific document content
  const loadDocumentContent = async (docId: string) => {
    try {
      const response = await fetch(`/api/documentation/persistent/${docId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to load document: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setSelectedFile(data.data);
      } else {
        throw new Error(data.error || 'Failed to load document');
      }

    } catch (err) {
      console.error('Error loading document content:', err);
      setError(err instanceof Error ? err.message : 'Failed to load document');
    }
  };

  // Search documentation
  const searchDocumentation = async (query: string) => {
    if (!query.trim()) {
      loadDocumentationFiles();
      return;
    }

    try {
      const response = await fetch(`/api/documentation/search?query=${encodeURIComponent(query)}`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          // Convert search results to file format
          const searchFiles = data.data.results.map((result: any) => ({
            id: result.path,
            title: result.title,
            path: result.path,
            category: result.category,
            size: 0,
            created_at: '',
            updated_at: '',
            generated_by_agent: false,
            backend_stored: false,
            temporary_copy: true
          }));
          setFiles(searchFiles);
        }
      }
    } catch (err) {
      console.error('Error searching documentation:', err);
    }
  };

  useEffect(() => {
    loadDocumentationFiles();
    loadCategories();
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      searchDocumentation(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Filter files by category
  const filteredFiles = selectedCategory === 'All' 
    ? files 
    : files.filter(file => file.category === selectedCategory);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading documentation...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <div className="max-w-md">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-red-800 mb-2">Error Loading Documentation</h2>
            <p className="text-red-600 mb-4">{error}</p>
            <button
              onClick={() => {
                setError(null);
                loadDocumentationFiles();
              }}
              className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-1/3 bg-gray-50 border-r overflow-y-auto">
        <div className="p-4">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">Documentation</h1>
          
          {/* Search */}
          <div className="mb-4">
            <input
              type="text"
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Categories */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Categories</h3>
            <div className="space-y-1">
              {categories.map(category => (
                <button
                  key={category.name}
                  onClick={() => setSelectedCategory(category.name)}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                    selectedCategory === category.name
                      ? 'bg-blue-100 text-blue-800'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {category.name} {category.count > 0 && `(${category.count})`}
                </button>
              ))}
            </div>
          </div>

          {/* Files list */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">
              Documents ({filteredFiles.length})
            </h3>
            <div className="space-y-1">
              {filteredFiles.map(file => (
                <button
                  key={file.id}
                  onClick={() => file.backend_stored ? loadDocumentContent(file.id) : null}
                  className={`w-full text-left p-3 rounded-md border transition-colors ${
                    selectedFile?.id === file.id
                      ? 'bg-blue-50 border-blue-200'
                      : 'bg-white border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium text-gray-800">{file.title}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {file.category}
                    {file.generated_by_agent && (
                      <span className="ml-2 bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
                        AI Generated
                      </span>
                    )}
                    {file.backend_stored && (
                      <span className="ml-2 bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        Persistent
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-y-auto">
        {selectedFile ? (
          <div className="p-6">
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <h1 className="text-3xl font-bold text-gray-800">{selectedFile.title}</h1>
                <div className="flex items-center space-x-2">
                  {selectedFile.backend_stored && (
                    <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                      Persistent
                    </span>
                  )}
                  {selectedFile.tags?.map(tag => (
                    <span key={tag} className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <div className="text-sm text-gray-500">
                Category: {selectedFile.category} | 
                Last modified: {new Date(selectedFile.modified).toLocaleDateString()}
                {selectedFile.versions && selectedFile.versions.length > 0 && (
                  <span> | Version: {selectedFile.versions[0].version_number}</span>
                )}
              </div>
            </div>
            
            {/* Content */}
            <div className="prose max-w-none">
              {selectedFile.html_content ? (
                <div dangerouslySetInnerHTML={{ __html: selectedFile.html_content }} />
              ) : (
                <pre className="whitespace-pre-wrap font-sans">{selectedFile.content}</pre>
              )}
            </div>

            {/* Version history */}
            {selectedFile.versions && selectedFile.versions.length > 1 && (
              <div className="mt-8 border-t pt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Version History</h3>
                <div className="space-y-2">
                  {selectedFile.versions.map(version => (
                    <div key={version.version_number} className="bg-gray-50 p-3 rounded">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Version {version.version_number}</span>
                        <span className="text-sm text-gray-500">
                          {new Date(version.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      {version.change_summary && (
                        <p className="text-sm text-gray-600 mt-1">{version.change_summary}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <h2 className="text-xl font-semibold text-gray-600 mb-2">
                Select a document to view
              </h2>
              <p className="text-gray-500">
                Choose a document from the sidebar to view its content
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentationPageV2;
