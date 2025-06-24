/**
 * Startup Configuration Page
 * Service configuration and profile management interface
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Settings, 
  Plus, 
  Edit, 
  Trash2, 
  Copy,
  Save,
  RefreshCw,
  FileText,
  Database,
  Server,
  Zap
} from 'lucide-react';
import { useStartupService } from '@/stores/appStore';
import { useConfigurations, useProfiles } from '@/hooks/useStartupService';
import { ConfigurationManager } from '@/components/startup';

const StartupConfigurationPage: React.FC = () => {
  const { startupService } = useStartupService();
  const { data: configurations, isLoading: configsLoading, refetch: refetchConfigs } = useConfigurations();
  const { data: profiles, isLoading: profilesLoading, refetch: refetchProfiles } = useProfiles();
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null);

  const getServiceTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'postgresql':
        return <Database className="h-4 w-4" />;
      case 'redis':
        return <Server className="h-4 w-4" />;
      case 'ollama':
        return <Zap className="h-4 w-4" />;
      default:
        return <Settings className="h-4 w-4" />;
    }
  };

  const getServiceTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'postgresql':
        return 'bg-blue-500';
      case 'redis':
        return 'bg-red-500';
      case 'ollama':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center">
              <Settings className="h-8 w-8 mr-3 text-purple-500" />
              Configuration Management
            </h1>
            <p className="text-muted-foreground mt-2">
              Manage service configurations and startup profiles
            </p>
          </div>
          
          <div className="flex space-x-2">
            <Button variant="outline" onClick={() => {
              refetchConfigs();
              refetchProfiles();
            }}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Configuration
            </Button>
          </div>
        </div>

        {/* Configuration Manager Component */}
        <ConfigurationManager
          configurations={configurations || []}
          profiles={profiles || []}
          onSaveConfiguration={(config) => {
            console.log('Save configuration:', config);
          }}
          onDeleteConfiguration={(configId) => {
            console.log('Delete configuration:', configId);
          }}
          onSaveProfile={(profile) => {
            console.log('Save profile:', profile);
          }}
          onDeleteProfile={(profileId) => {
            console.log('Delete profile:', profileId);
          }}
        />

        {/* Legacy Configuration Tabs */}
        <Tabs defaultValue="configurations" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="configurations">Service Configurations</TabsTrigger>
            <TabsTrigger value="profiles">Startup Profiles</TabsTrigger>
          </TabsList>

          {/* Service Configurations Tab */}
          <TabsContent value="configurations" className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <Settings className="h-8 w-8 text-blue-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Total Configs</p>
                      <p className="text-2xl font-bold">{configurations?.length || 0}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <Database className="h-8 w-8 text-green-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Active Configs</p>
                      <p className="text-2xl font-bold">
                        {configurations?.filter(c => c.is_active).length || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <Server className="h-8 w-8 text-orange-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Service Types</p>
                      <p className="text-2xl font-bold">
                        {new Set(configurations?.map(c => c.service_type)).size || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <FileText className="h-8 w-8 text-purple-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Environments</p>
                      <p className="text-2xl font-bold">
                        {new Set(configurations?.map(c => c.environment)).size || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Configurations List */}
            <Card>
              <CardHeader>
                <CardTitle>Service Configurations</CardTitle>
                <CardDescription>
                  Manage individual service configurations and settings
                </CardDescription>
              </CardHeader>
              <CardContent>
                {configsLoading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                    <p className="mt-2 text-muted-foreground">Loading configurations...</p>
                  </div>
                ) : configurations && configurations.length > 0 ? (
                  <div className="space-y-4">
                    {configurations.map((config) => (
                      <div
                        key={config.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          selectedConfig === config.id ? 'border-primary bg-primary/5' : 'hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedConfig(
                          selectedConfig === config.id ? null : config.id
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            {getServiceTypeIcon(config.service_type)}
                            <div>
                              <h3 className="font-semibold">{config.service_name}</h3>
                              <p className="text-sm text-muted-foreground">
                                {config.service_type} • {config.environment} • v{config.version}
                              </p>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {config.is_active && (
                              <Badge variant="default" className="bg-green-500">
                                Active
                              </Badge>
                            )}
                            <Badge variant="outline" className={getServiceTypeColor(config.service_type)}>
                              {config.service_type}
                            </Badge>
                            
                            <div className="flex space-x-1">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle edit action
                                }}
                              >
                                <Edit className="h-3 w-3" />
                              </Button>
                              
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle copy action
                                }}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                              
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle delete action
                                }}
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        </div>

                        {/* Expanded Details */}
                        {selectedConfig === config.id && (
                          <div className="mt-4 pt-4 border-t space-y-3">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Created:</span>
                                <p className="text-muted-foreground">
                                  {new Date(config.created_at).toLocaleDateString()}
                                </p>
                              </div>
                              
                              <div>
                                <span className="font-medium">Last Updated:</span>
                                <p className="text-muted-foreground">
                                  {new Date(config.updated_at).toLocaleDateString()}
                                </p>
                              </div>
                            </div>

                            <div>
                              <span className="font-medium">Configuration:</span>
                              <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-auto max-h-32">
                                {JSON.stringify(config.configuration, null, 2)}
                              </pre>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Settings className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                    <h3 className="text-lg font-semibold mb-2">No Configurations Found</h3>
                    <p className="text-gray-600 mb-4">
                      Create your first service configuration to get started.
                    </p>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Configuration
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Startup Profiles Tab */}
          <TabsContent value="profiles" className="space-y-6">
            {/* Profile Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <FileText className="h-8 w-8 text-blue-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Total Profiles</p>
                      <p className="text-2xl font-bold">{profiles?.length || 0}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <Zap className="h-8 w-8 text-green-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Active Profiles</p>
                      <p className="text-2xl font-bold">
                        {profiles?.filter(p => p.is_active).length || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <Save className="h-8 w-8 text-orange-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Default Profile</p>
                      <p className="text-2xl font-bold">
                        {profiles?.filter(p => p.is_default).length || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center">
                    <RefreshCw className="h-8 w-8 text-purple-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-muted-foreground">Usage Count</p>
                      <p className="text-2xl font-bold">
                        {profiles?.reduce((sum, p) => sum + p.usage_count, 0) || 0}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Profiles List */}
            <Card>
              <CardHeader>
                <CardTitle>Startup Profiles</CardTitle>
                <CardDescription>
                  Manage startup profiles and environment configurations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {profilesLoading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                    <p className="mt-2 text-muted-foreground">Loading profiles...</p>
                  </div>
                ) : profiles && profiles.length > 0 ? (
                  <div className="space-y-4">
                    {profiles.map((profile) => (
                      <div
                        key={profile.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          selectedProfile === profile.id ? 'border-primary bg-primary/5' : 'hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedProfile(
                          selectedProfile === profile.id ? null : profile.id
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <FileText className="h-5 w-5 text-blue-500" />
                            <div>
                              <h3 className="font-semibold">{profile.profile_name}</h3>
                              <p className="text-sm text-muted-foreground">
                                {profile.startup_sequence.length} services • Used {profile.usage_count} times
                              </p>
                              {profile.description && (
                                <p className="text-xs text-muted-foreground mt-1">
                                  {profile.description}
                                </p>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {profile.is_default && (
                              <Badge variant="default" className="bg-blue-500">
                                Default
                              </Badge>
                            )}
                            {profile.is_active && (
                              <Badge variant="default" className="bg-green-500">
                                Active
                              </Badge>
                            )}
                            
                            <div className="flex space-x-1">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle edit action
                                }}
                              >
                                <Edit className="h-3 w-3" />
                              </Button>
                              
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle copy action
                                }}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        </div>

                        {/* Expanded Details */}
                        {selectedProfile === profile.id && (
                          <div className="mt-4 pt-4 border-t space-y-3">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Profile Type:</span>
                                <p className="text-muted-foreground">{profile.profile_type}</p>
                              </div>
                              
                              <div>
                                <span className="font-medium">Last Used:</span>
                                <p className="text-muted-foreground">
                                  {profile.last_used ? new Date(profile.last_used).toLocaleDateString() : 'Never'}
                                </p>
                              </div>
                            </div>

                            <div>
                              <span className="font-medium">Startup Sequence:</span>
                              <div className="mt-1 space-y-1">
                                {profile.startup_sequence.map((service, index) => (
                                  <Badge key={index} variant="secondary" className="mr-1">
                                    {index + 1}. {service}
                                  </Badge>
                                ))}
                              </div>
                            </div>

                            {profile.tags && profile.tags.length > 0 && (
                              <div>
                                <span className="font-medium">Tags:</span>
                                <div className="mt-1 space-y-1">
                                  {profile.tags.map((tag, index) => (
                                    <Badge key={index} variant="outline" className="mr-1">
                                      {tag}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <FileText className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                    <h3 className="text-lg font-semibold mb-2">No Profiles Found</h3>
                    <p className="text-gray-600 mb-4">
                      Create your first startup profile to save configuration sets.
                    </p>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Profile
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default StartupConfigurationPage;
