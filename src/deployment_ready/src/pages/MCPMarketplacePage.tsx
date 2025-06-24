import React, { useState, useEffect } from 'react';
import { Package, Download, CheckCircle, AlertCircle, ExternalLink } from 'lucide-react';

interface MCPServer {
  id: string;
  name: string;
  description: string;
  status: 'available' | 'installed' | 'running' | 'error';
  capabilities: string[];
  url?: string;
  documentation?: string;
}

export const MCPMarketplacePage: React.FC = () => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [filter, setFilter] = useState<'all' | 'installed' | 'available'>('all');

  useEffect(() => {
    // Mock MCP servers data - in real implementation, this would come from the backend
    const mockServers: MCPServer[] = [
      {
        id: 'cloudflare-mcp',
        name: 'Cloudflare MCP Server',
        description: 'Official Cloudflare MCP server for tunnel and infrastructure management',
        status: 'installed',
        capabilities: ['tunnels', 'dns', 'workers', 'pages'],
        url: 'https://github.com/cloudflare/mcp-server-cloudflare',
        documentation: 'https://github.com/cloudflare/mcp-server-cloudflare#readme'
      },
      {
        id: 'filesystem-mcp',
        name: 'Filesystem MCP Server',
        description: 'File system operations and management',
        status: 'running',
        capabilities: ['file_read', 'file_write', 'directory_list'],
        url: 'https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem',
        documentation: 'https://modelcontextprotocol.io/servers/filesystem'
      },
      {
        id: 'postgresql-mcp',
        name: 'PostgreSQL MCP Server',
        description: 'Database operations and query execution',
        status: 'running',
        capabilities: ['database_query', 'schema_read', 'data_write'],
        url: 'https://github.com/modelcontextprotocol/servers/tree/main/src/postgres',
        documentation: 'https://modelcontextprotocol.io/servers/postgres'
      },
      {
        id: 'github-mcp',
        name: 'GitHub MCP Server',
        description: 'GitHub repository and issue management',
        status: 'available',
        capabilities: ['repo_read', 'issue_create', 'pr_create'],
        url: 'https://github.com/modelcontextprotocol/servers/tree/main/src/github',
        documentation: 'https://modelcontextprotocol.io/servers/github'
      },
      {
        id: 'brave-search-mcp',
        name: 'Brave Search MCP Server',
        description: 'Web search capabilities via Brave Search API',
        status: 'available',
        capabilities: ['web_search', 'search_results'],
        url: 'https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search',
        documentation: 'https://modelcontextprotocol.io/servers/brave-search'
      }
    ];

    setServers(mockServers);
  }, []);

  const filteredServers = servers.filter(server => {
    if (filter === 'all') return true;
    if (filter === 'installed') return server.status === 'installed' || server.status === 'running';
    if (filter === 'available') return server.status === 'available';
    return true;
  });

  const getStatusIcon = (status: MCPServer['status']) => {
    switch (status) {
      case 'running':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'installed':
        return <CheckCircle className="w-5 h-5 text-blue-500" />;
      case 'available':
        return <Download className="w-5 h-5 text-muted-foreground" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
    }
  };

  const getStatusText = (status: MCPServer['status']) => {
    switch (status) {
      case 'running':
        return 'Running';
      case 'installed':
        return 'Installed';
      case 'available':
        return 'Available';
      case 'error':
        return 'Error';
    }
  };

  const getStatusColor = (status: MCPServer['status']) => {
    switch (status) {
      case 'running':
        return 'text-green-500 bg-green-50 border-green-200';
      case 'installed':
        return 'text-blue-500 bg-blue-50 border-blue-200';
      case 'available':
        return 'text-muted-foreground bg-muted border-border';
      case 'error':
        return 'text-red-500 bg-red-50 border-red-200';
    }
  };

  const handleInstall = (serverId: string) => {
    // Mock installation - in real implementation, this would call the backend
    setServers(prev => prev.map(server => 
      server.id === serverId 
        ? { ...server, status: 'installed' as const }
        : server
    ));
  };

  const handleStart = (serverId: string) => {
    // Mock start - in real implementation, this would call the backend
    setServers(prev => prev.map(server => 
      server.id === serverId 
        ? { ...server, status: 'running' as const }
        : server
    ));
  };

  const handleStop = (serverId: string) => {
    // Mock stop - in real implementation, this would call the backend
    setServers(prev => prev.map(server => 
      server.id === serverId 
        ? { ...server, status: 'installed' as const }
        : server
    ));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <Package className="w-8 h-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold text-foreground">MCP Marketplace</h1>
          <p className="text-muted-foreground">Discover and manage Model Context Protocol servers</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <span className="text-sm font-medium text-foreground">Filter:</span>
        <div className="flex space-x-2">
          {[
            { id: 'all', label: 'All Servers' },
            { id: 'installed', label: 'Installed' },
            { id: 'available', label: 'Available' }
          ].map((filterOption) => (
            <button
              key={filterOption.id}
              onClick={() => setFilter(filterOption.id as any)}
              className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                filter === filterOption.id
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              {filterOption.label}
            </button>
          ))}
        </div>
      </div>

      {/* Server Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredServers.map((server) => (
          <div key={server.id} className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="font-semibold text-foreground mb-1">{server.name}</h3>
                <p className="text-sm text-muted-foreground">{server.description}</p>
              </div>
              <div className="ml-4">
                {getStatusIcon(server.status)}
              </div>
            </div>

            {/* Status Badge */}
            <div className="mb-4">
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(server.status)}`}>
                {getStatusText(server.status)}
              </span>
            </div>

            {/* Capabilities */}
            <div className="mb-4">
              <p className="text-sm font-medium text-foreground mb-2">Capabilities:</p>
              <div className="flex flex-wrap gap-1">
                {server.capabilities.map((capability) => (
                  <span
                    key={capability}
                    className="inline-flex items-center px-2 py-1 rounded bg-muted text-xs text-muted-foreground"
                  >
                    {capability}
                  </span>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between">
              <div className="flex space-x-2">
                {server.status === 'available' && (
                  <button
                    onClick={() => handleInstall(server.id)}
                    className="flex items-center space-x-1 px-3 py-1 bg-primary text-primary-foreground rounded text-sm hover:bg-primary/90 transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    <span>Install</span>
                  </button>
                )}
                
                {server.status === 'installed' && (
                  <button
                    onClick={() => handleStart(server.id)}
                    className="flex items-center space-x-1 px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600 transition-colors"
                  >
                    <span>Start</span>
                  </button>
                )}
                
                {server.status === 'running' && (
                  <button
                    onClick={() => handleStop(server.id)}
                    className="flex items-center space-x-1 px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600 transition-colors"
                  >
                    <span>Stop</span>
                  </button>
                )}
              </div>

              {/* Links */}
              <div className="flex space-x-2">
                {server.url && (
                  <a
                    href={server.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1 text-muted-foreground hover:text-foreground transition-colors"
                    title="View Source"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredServers.length === 0 && (
        <div className="text-center py-12">
          <Package className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
          <p className="text-lg font-medium text-foreground">No servers found</p>
          <p className="text-muted-foreground">Try adjusting your filter or check back later</p>
        </div>
      )}
    </div>
  );
};