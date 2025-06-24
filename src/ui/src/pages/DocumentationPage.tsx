import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

/**
 * DocumentationPage Component
 * 
 * Serves the built VitePress documentation as part of the main PyGent Factory UI.
 * This component provides seamless integration between the React app and static documentation.
 */
export const DocumentationPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Extract the documentation path from the current route
  const docPath = location.pathname.replace('/docs', '') || '/';
  const fullDocUrl = `/docs${docPath}`;

  useEffect(() => {
    // Check if documentation is available
    const checkDocumentation = async () => {
      try {
        const response = await fetch('/docs/manifest.json');
        if (!response.ok) {
          throw new Error('Documentation not available');
        }
        setIsLoading(false);
      } catch (err) {
        setError('Documentation is not available. Please build the documentation first.');
        setIsLoading(false);
      }
    };

    checkDocumentation();
  }, []);

  const handleIframeLoad = () => {
    setIsLoading(false);
  };

  const handleIframeError = () => {
    setError('Failed to load documentation page');
    setIsLoading(false);
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <div className="max-w-md">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Documentation Unavailable</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <div className="space-y-2 text-sm text-gray-500">
            <p>To build the documentation:</p>
            <code className="block bg-gray-100 p-2 rounded">
              cd src/docs && npm run build:hybrid
            </code>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading documentation...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full relative">
      {/* Documentation iframe for seamless integration */}
      <iframe
        src={fullDocUrl}
        className="w-full h-full border-0"
        title="PyGent Factory Documentation"
        onLoad={handleIframeLoad}
        onError={handleIframeError}
        style={{
          minHeight: 'calc(100vh - 120px)', // Account for header/navigation
        }}
      />
      
      {/* Overlay for loading state */}
      {isLoading && (
        <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-600">Loading...</p>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Alternative implementation using direct HTML injection
 * This approach provides better integration but requires more complex setup
 */
export const DocumentationPageDirect: React.FC = () => {
  const location = useLocation();
  const [htmlContent, setHtmlContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const docPath = location.pathname.replace('/docs', '') || '/';

  useEffect(() => {
    const loadDocumentation = async () => {
      try {
        setIsLoading(true);
        
        // Determine the HTML file to load
        let htmlFile = 'index.html';
        if (docPath !== '/') {
          htmlFile = `${docPath.replace(/^\//, '').replace(/\/$/, '')}.html`;
        }

        const response = await fetch(`/docs/${htmlFile}`);
        if (!response.ok) {
          throw new Error(`Documentation page not found: ${htmlFile}`);
        }

        const html = await response.text();
        
        // Process the HTML to make it work within the React app
        const processedHtml = html
          .replace(/href="\//g, 'href="/docs/')
          .replace(/src="\//g, 'src="/docs/')
          .replace(/<base[^>]*>/g, ''); // Remove base tag that might conflict

        setHtmlContent(processedHtml);
        setIsLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load documentation');
        setIsLoading(false);
      }
    };

    loadDocumentation();
  }, [docPath]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Documentation Error</h2>
        <p className="text-gray-600">{error}</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div 
      className="h-full w-full overflow-auto"
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
};

export default DocumentationPage;
