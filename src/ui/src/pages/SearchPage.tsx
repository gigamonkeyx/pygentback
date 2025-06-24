import React, { useState } from 'react';
import { Search, Filter, Database, Zap, FileText, Clock } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

export const SearchPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults] = useState([
    {
      id: 1,
      title: 'Advanced AI Reasoning Techniques',
      content: 'Comprehensive guide to implementing Tree of Thought reasoning in AI systems...',
      score: 0.95,
      source: 'research_papers.pdf',
      timestamp: '2024-01-15'
    },
    {
      id: 2,
      title: 'Vector Database Optimization',
      content: 'Best practices for optimizing vector databases for high-performance search...',
      score: 0.87,
      source: 'technical_docs.md',
      timestamp: '2024-01-14'
    },
    {
      id: 3,
      title: 'GPU Acceleration for ML Workloads',
      content: 'Implementation strategies for leveraging GPU acceleration in machine learning...',
      score: 0.82,
      source: 'ml_handbook.pdf',
      timestamp: '2024-01-13'
    }
  ]);

  const handleSearch = () => {
    console.log('Searching for:', searchQuery);
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <Search className="h-8 w-8 text-purple-500" />
        <div>
          <h1 className="text-3xl font-bold">Vector Search</h1>
          <p className="text-muted-foreground">
            GPU-accelerated vector search and document retrieval
          </p>
        </div>
      </div>

      {/* Search Interface */}
      <Card>
        <CardHeader>
          <CardTitle>Search Documents</CardTitle>
          <CardDescription>
            Search through your document collection using semantic vector search
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2 mb-4">
            <Input
              placeholder="Enter your search query..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <Button onClick={handleSearch}>
              <Search className="h-4 w-4 mr-2" />
              Search
            </Button>
            <Button variant="outline">
              <Filter className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Search Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1,247</div>
            <p className="text-xs text-muted-foreground">Indexed documents</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Vector Dimensions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1,536</div>
            <p className="text-xs text-muted-foreground">Embedding dimensions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Search Speed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12ms</div>
            <p className="text-xs text-muted-foreground">Average query time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">GPU Utilization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">78%</div>
            <p className="text-xs text-muted-foreground">Current GPU usage</p>
          </CardContent>
        </Card>
      </div>

      {/* Search Results */}
      <Card>
        <CardHeader>
          <CardTitle>Search Results</CardTitle>
          <CardDescription>
            {searchResults.length} results found
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {searchResults.map((result) => (
              <div key={result.id} className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-lg">{result.title}</h3>
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Zap className="h-3 w-3" />
                    <span>{(result.score * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <p className="text-muted-foreground mb-3">{result.content}</p>
                <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                  <div className="flex items-center space-x-1">
                    <FileText className="h-3 w-3" />
                    <span>{result.source}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{result.timestamp}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Index Management */}
      <Card>
        <CardHeader>
          <CardTitle>Index Management</CardTitle>
          <CardDescription>
            Manage your vector search indexes and embeddings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-blue-500" />
              <div>
                <p className="font-medium">Primary Index</p>
                <p className="text-sm text-muted-foreground">Last updated: 2 hours ago</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm">Rebuild Index</Button>
              <Button variant="outline" size="sm">Export</Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
