import React, { useState, useEffect } from 'react';
import { Package, Download, Play, Square, Settings, CheckCircle, AlertCircle, RefreshCw, Server, Database, Globe, Code, Palette, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface MCPServer {
  id: string;
  name: string;
  description: string;
  category: string;
  status: 'installed' | 'available' | 'running' | 'stopped' | 'error';
  version: string;
  capabilities: string[];
  author?: string;
  verified?: boolean;
  tools?: string[];
}

interface DiscoveryStatus {
  discovery_enabled: boolean;
  status: string;
  results: {
    cache_loaded: boolean;
    servers_discovered: number;
    priority_servers_registered: number;
    additional_servers_registered: number;
    total_servers_registered: number;
    startup_time_ms: number;
    success: boolean;
  };
  summary: {
    servers_discovered: number;
    servers_registered: number;
    priority_servers: number;
    startup_time_ms: number;
  };
}

interface DiscoveredServers {
  total_discovered: number;
  categories: Record<string, MCPServer[]>;
  cache_file: string;
}

export const MCPMarketplacePage: React.FC = () => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [discoveryStatus, setDiscoveryStatus] = useState<DiscoveryStatus | null>(null);
  const [discoveredServers, setDiscoveredServers] = useState<DiscoveredServers | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [installingServers, setInstallingServers] = useState<Set<string>>(new Set());
  const [installationStatus, setInstallationStatus] = useState<Record<string, any>>({});

  const fetchDiscoveryData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch discovery status
      const statusResponse = await fetch('/api/v1/mcp/discovery/status');
      if (!statusResponse.ok) throw new Error('Failed to fetch discovery status');
      const statusData = await statusResponse.json();
      setDiscoveryStatus(statusData);

      // Fetch discovered servers
      const serversResponse = await fetch('/api/v1/mcp/discovery/servers');
      if (!serversResponse.ok) throw new Error('Failed to fetch discovered servers');
      const serversData = await serversResponse.json();
      setDiscoveredServers(serversData);

      // Transform discovered servers into UI format
      const allServers: MCPServer[] = [];
      Object.entries(serversData.categories).forEach(([category, categoryServers]) => {
        categoryServers.forEach((server: any) => {
          allServers.push({
            id: server.name,
            name: server.name,
            description: server.description || 'No description available',
            category: category,
            status: getServerStatus(server.name, statusData),
            version: '1.0.0', // Default version
            capabilities: server.capabilities || [],
            author: server.author || 'Unknown',
            verified: server.verified || false,
            tools: server.tools || []
          });
        });
      });

      setServers(allServers);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load MCP data');
    } finally {
      setLoading(false);
    }
  };

  const getServerStatus = (serverName: string, statusData: DiscoveryStatus): MCPServer['status'] => {
    // All discovered servers are available but not installed
    // They need actual installation before they can be used
    return 'available';
  };

  const installServer = async (serverName: string) => {
    try {
      setInstallingServers(prev => new Set([...prev, serverName]));

      const response = await fetch('/api/v1/mcp/servers/install', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          server_name: serverName,
          source_type: 'npm',
          auto_start: true
        })
      });

      if (!response.ok) {
        throw new Error('Failed to start installation');
      }

      const result = await response.json();

      // Start polling for installation status
      pollInstallationStatus(serverName);

    } catch (err) {
      console.error('Installation failed:', err);
      setInstallingServers(prev => {
        const newSet = new Set(prev);
        newSet.delete(serverName);
        return newSet;
      });
      alert(`Failed to install ${serverName}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const pollInstallationStatus = async (serverName: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/mcp/servers/install/${serverName}/status`);
        if (response.ok) {
          const status = await response.json();
          setInstallationStatus(prev => ({
            ...prev,
            [serverName]: status
          }));

          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(pollInterval);
            setInstallingServers(prev => {
              const newSet = new Set(prev);
              newSet.delete(serverName);
              return newSet;
            });

            if (status.status === 'completed') {
              // Refresh the discovery data to show updated server status
              fetchDiscoveryData();
            }
          }
        }
      } catch (err) {
        console.error('Failed to poll installation status:', err);
        clearInterval(pollInterval);
        setInstallingServers(prev => {
          const newSet = new Set(prev);
          newSet.delete(serverName);
          return newSet;
        });
      }
    }, 2000); // Poll every 2 seconds
  };

  useEffect(() => {
    fetchDiscoveryData();
  }, []);

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'development':
        return <Code className="h-4 w-4" />;
      case 'database':
        return <Database className="h-4 w-4" />;
      case 'nlp':
        return <Globe className="h-4 w-4" />;
      case 'web_ui':
        return <Palette className="h-4 w-4" />;
      case 'academic_research':
        return <BarChart3 className="h-4 w-4" />;
      case 'coding':
        return <Code className="h-4 w-4" />;
      default:
        return <Server className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: MCPServer['status']) => {
    switch (status) {
      case 'running':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'installed':
        return <CheckCircle className="h-4 w-4 text-blue-500" />;
      case 'stopped':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Package className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: MCPServer['status']) => {
    switch (status) {
      case 'running':
        return 'text-green-600 bg-green-50';
      case 'installed':
        return 'text-blue-600 bg-blue-50';
      case 'stopped':
        return 'text-yellow-600 bg-yellow-50';
      case 'error':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-indigo-500" />
          <span className="ml-2 text-lg">Loading MCP servers...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <AlertCircle className="h-8 w-8 text-red-500" />
          <div className="ml-2">
            <p className="text-lg font-medium text-red-600">Error loading MCP data</p>
            <p className="text-sm text-red-500">{error}</p>
            <Button onClick={fetchDiscoveryData} className="mt-2">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Package className="h-8 w-8 text-indigo-500" />
          <div>
            <h1 className="text-3xl font-bold">MCP Marketplace</h1>
            <p className="text-muted-foreground">
              Model Context Protocol server management and marketplace
            </p>
          </div>
        </div>
        <Button onClick={fetchDiscoveryData} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Discovery Statistics */}
      {discoveryStatus && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Server className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Found</p>
                  <p className="text-2xl font-bold">{discoveryStatus.results.servers_discovered}</p>
                  <p className="text-xs text-muted-foreground">servers discovered</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Package className="h-5 w-5 text-orange-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Cataloged</p>
                  <p className="text-2xl font-bold">{discoveryStatus.results.total_servers_registered}</p>
                  <p className="text-xs text-muted-foreground">in system registry</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Installed</p>
                  <p className="text-2xl font-bold">0</p>
                  <p className="text-xs text-muted-foreground">ready to use</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <RefreshCw className="h-5 w-5 text-purple-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Discovery</p>
                  <p className="text-2xl font-bold">{discoveryStatus.results.startup_time_ms}ms</p>
                  <p className="text-xs text-muted-foreground">scan time</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Server Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {servers.map((server) => (
          <Card key={server.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getCategoryIcon(server.category)}
                  <CardTitle className="text-lg">{server.name}</CardTitle>
                  {server.verified && (
                    <Badge variant="secondary" className="text-xs">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Verified
                    </Badge>
                  )}
                </div>
                {getStatusIcon(server.status)}
              </div>
              <CardDescription>{server.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Category</span>
                  <Badge variant="outline" className="text-xs">
                    {server.category.replace('_', ' ')}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(server.status)}`}>
                    {server.status}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Author</span>
                  <span className="text-sm">{server.author}</span>
                </div>

                <div>
                  <span className="text-sm text-muted-foreground">Capabilities</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {server.capabilities.slice(0, 2).map((cap) => (
                      <Badge key={cap} variant="secondary" className="text-xs">
                        {cap}
                      </Badge>
                    ))}
                    {server.capabilities.length > 2 && (
                      <Badge variant="secondary" className="text-xs">
                        +{server.capabilities.length - 2} more
                      </Badge>
                    )}
                  </div>
                </div>

                {server.tools && server.tools.length > 0 && (
                  <div>
                    <span className="text-sm text-muted-foreground">Tools</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {server.tools.slice(0, 2).map((tool) => (
                        <Badge key={tool} variant="outline" className="text-xs">
                          {tool}
                        </Badge>
                      ))}
                      {server.tools.length > 2 && (
                        <Badge variant="outline" className="text-xs">
                          +{server.tools.length - 2} more
                        </Badge>
                      )}
                    </div>
                  </div>
                )}

                <div className="flex space-x-2 pt-2">
                  {server.status === 'available' && !installingServers.has(server.id) && (
                    <Button
                      size="sm"
                      className="flex-1"
                      onClick={() => installServer(server.id)}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      Install
                    </Button>
                  )}
                  {installingServers.has(server.id) && (
                    <Button size="sm" className="flex-1" disabled>
                      <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                      {installationStatus[server.id]?.progress || 0}%
                    </Button>
                  )}
                  {server.status === 'installed' && (
                    <Button size="sm" variant="outline" className="flex-1" disabled>
                      <Play className="h-3 w-3 mr-1" />
                      Start
                    </Button>
                  )}
                  {server.status === 'running' && (
                    <Button size="sm" variant="outline" className="flex-1" disabled>
                      <Square className="h-3 w-3 mr-1" />
                      Stop
                    </Button>
                  )}
                  {server.status === 'stopped' && (
                    <Button size="sm" variant="outline" className="flex-1" disabled>
                      <Play className="h-3 w-3 mr-1" />
                      Start
                    </Button>
                  )}
                  {server.status === 'error' && (
                    <Button size="sm" variant="outline" className="flex-1" disabled>
                      <AlertCircle className="h-3 w-3 mr-1" />
                      Error
                    </Button>
                  )}
                  <Button size="sm" variant="ghost" disabled>
                    <Settings className="h-3 w-3" />
                  </Button>
                </div>

                {/* Installation Progress */}
                {installingServers.has(server.id) && installationStatus[server.id] && (
                  <div className="mt-2 p-2 bg-blue-50 rounded-md">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-blue-600 font-medium">Installing...</span>
                      <span className="text-blue-500">{installationStatus[server.id].progress}%</span>
                    </div>
                    <div className="mt-1 w-full bg-blue-200 rounded-full h-1">
                      <div
                        className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                        style={{ width: `${installationStatus[server.id].progress}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-blue-600 mt-1">{installationStatus[server.id].message}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Category Summary */}
      {discoveredServers && (
        <Card>
          <CardHeader>
            <CardTitle>Server Categories</CardTitle>
            <CardDescription>
              MCP servers organized by functionality and purpose
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(discoveredServers.categories).map(([category, categoryServers]) => (
                <div key={category} className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
                  {getCategoryIcon(category)}
                  <div>
                    <p className="font-medium">{category.replace('_', ' ')}</p>
                    <p className="text-sm text-muted-foreground">{categoryServers.length} servers</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Installation Notice */}
      <Card>
        <CardHeader>
          <CardTitle>Installation System</CardTitle>
          <CardDescription>
            MCP server discovery and installation capabilities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Package className="h-12 w-12 mx-auto mb-4 text-green-500" />
            <p className="text-lg font-medium mb-2">Ready to Install MCP Servers</p>
            <p className="text-sm text-muted-foreground mb-4">
              <strong>19 servers discovered</strong> and <strong>8 cataloged</strong> in the system registry.
              Click the <strong>"Install"</strong> button on any server card to automatically install and configure it.
              Installation includes npm package download, dependency resolution, and server registration.
            </p>
            <div className="flex justify-center space-x-2">
              <Badge variant="secondary">Discovery: ✓ Active</Badge>
              <Badge variant="secondary">Installation: ✓ Ready</Badge>
              <Badge variant="outline">Auto-Configuration: ✓ Enabled</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
