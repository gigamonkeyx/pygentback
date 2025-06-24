/**
 * Configuration Manager Component
 * Advanced configuration and profile management with validation
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { 
  Settings, 
  Plus, 
  Edit, 
  Trash2, 
  Copy,
  Save,
  FileText,
  Database,
  Server,
  Zap,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Upload,
  Download
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ConfigurationManagerProps } from './types';
import { ServiceConfiguration, ConfigurationProfile } from '@/types';

const ConfigurationManager: React.FC<ConfigurationManagerProps> = ({
  configurations,
  profiles,
  onSaveConfiguration,
  onDeleteConfiguration,
  onSaveProfile,
  onDeleteProfile,
  className
}) => {
  const [selectedConfig, setSelectedConfig] = useState<ServiceConfiguration | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<ConfigurationProfile | null>(null);
  const [editingConfig, setEditingConfig] = useState(false);
  const [editingProfile, setEditingProfile] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Configuration form state
  const [configForm, setConfigForm] = useState({
    service_name: '',
    service_type: '',
    configuration: '{}',
    environment: 'development',
    version: '1.0.0'
  });

  // Profile form state
  const [profileForm, setProfileForm] = useState({
    profile_name: '',
    description: '',
    profile_type: 'standard',
    startup_sequence: [] as string[],
    environment_variables: '{}',
    tags: [] as string[]
  });

  const getServiceTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'postgresql':
        return <Database className="h-4 w-4 text-blue-500" />;
      case 'redis':
        return <Server className="h-4 w-4 text-red-500" />;
      case 'ollama':
        return <Zap className="h-4 w-4 text-purple-500" />;
      default:
        return <Settings className="h-4 w-4 text-gray-500" />;
    }
  };

  const validateConfiguration = (config: any): string[] => {
    const errors: string[] = [];
    
    if (!config.service_name?.trim()) {
      errors.push('Service name is required');
    }
    
    if (!config.service_type?.trim()) {
      errors.push('Service type is required');
    }
    
    try {
      JSON.parse(config.configuration);
    } catch (e) {
      errors.push('Configuration must be valid JSON');
    }
    
    return errors;
  };

  const validateProfile = (profile: any): string[] => {
    const errors: string[] = [];
    
    if (!profile.profile_name?.trim()) {
      errors.push('Profile name is required');
    }
    
    if (profile.startup_sequence.length === 0) {
      errors.push('At least one service must be in the startup sequence');
    }
    
    try {
      JSON.parse(profile.environment_variables);
    } catch (e) {
      errors.push('Environment variables must be valid JSON');
    }
    
    return errors;
  };

  const handleSaveConfiguration = () => {
    const errors = validateConfiguration(configForm);
    setValidationErrors(errors);
    
    if (errors.length === 0) {
      onSaveConfiguration({
        ...configForm,
        configuration: JSON.parse(configForm.configuration)
      });
      setEditingConfig(false);
      resetConfigForm();
    }
  };

  const handleSaveProfile = () => {
    const errors = validateProfile(profileForm);
    setValidationErrors(errors);
    
    if (errors.length === 0) {
      onSaveProfile({
        ...profileForm,
        environment_variables: JSON.parse(profileForm.environment_variables),
        services_config: {}
      });
      setEditingProfile(false);
      resetProfileForm();
    }
  };

  const resetConfigForm = () => {
    setConfigForm({
      service_name: '',
      service_type: '',
      configuration: '{}',
      environment: 'development',
      version: '1.0.0'
    });
    setValidationErrors([]);
  };

  const resetProfileForm = () => {
    setProfileForm({
      profile_name: '',
      description: '',
      profile_type: 'standard',
      startup_sequence: [],
      environment_variables: '{}',
      tags: []
    });
    setValidationErrors([]);
  };

  const loadConfigForEdit = (config: ServiceConfiguration) => {
    setConfigForm({
      service_name: config.service_name,
      service_type: config.service_type,
      configuration: JSON.stringify(config.configuration, null, 2),
      environment: config.environment,
      version: config.version
    });
    setSelectedConfig(config);
    setEditingConfig(true);
  };

  const loadProfileForEdit = (profile: ConfigurationProfile) => {
    setProfileForm({
      profile_name: profile.profile_name,
      description: profile.description || '',
      profile_type: profile.profile_type,
      startup_sequence: profile.startup_sequence,
      environment_variables: JSON.stringify(profile.environment_variables, null, 2),
      tags: profile.tags
    });
    setSelectedProfile(profile);
    setEditingProfile(true);
  };

  return (
    <div className={cn('space-y-6', className)}>
      <Tabs defaultValue="configurations" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="configurations">Service Configurations</TabsTrigger>
          <TabsTrigger value="profiles">Startup Profiles</TabsTrigger>
        </TabsList>

        {/* Service Configurations Tab */}
        <TabsContent value="configurations" className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Service Configurations</CardTitle>
                <CardDescription>
                  Manage individual service configurations and settings
                </CardDescription>
              </div>
              <Button onClick={() => setEditingConfig(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Configuration
              </Button>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {configurations.map((config) => (
                  <Card key={config.id} className="border-l-4 border-l-blue-500">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          {getServiceTypeIcon(config.service_type)}
                          <h4 className="font-semibold">{config.service_name}</h4>
                        </div>
                        {config.is_active && (
                          <Badge variant="default" className="bg-green-500">Active</Badge>
                        )}
                      </div>
                      
                      <div className="space-y-1 text-sm text-muted-foreground">
                        <p>Type: {config.service_type}</p>
                        <p>Environment: {config.environment}</p>
                        <p>Version: {config.version}</p>
                      </div>
                      
                      <div className="flex space-x-1 mt-3">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => loadConfigForEdit(config)}
                        >
                          <Edit className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            const newConfig = { ...config, service_name: `${config.service_name}_copy` };
                            loadConfigForEdit(newConfig);
                          }}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => onDeleteConfiguration(config.id)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Startup Profiles Tab */}
        <TabsContent value="profiles" className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Startup Profiles</CardTitle>
                <CardDescription>
                  Manage startup profiles and environment configurations
                </CardDescription>
              </div>
              <Button onClick={() => setEditingProfile(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Profile
              </Button>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {profiles.map((profile) => (
                  <Card key={profile.id} className="border-l-4 border-l-green-500">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-blue-500" />
                          <h4 className="font-semibold">{profile.profile_name}</h4>
                        </div>
                        <div className="flex space-x-1">
                          {profile.is_default && (
                            <Badge variant="default" className="bg-blue-500">Default</Badge>
                          )}
                          {profile.is_active && (
                            <Badge variant="default" className="bg-green-500">Active</Badge>
                          )}
                        </div>
                      </div>
                      
                      <div className="space-y-1 text-sm text-muted-foreground">
                        <p>{profile.description}</p>
                        <p>Services: {profile.startup_sequence.length}</p>
                        <p>Used: {profile.usage_count} times</p>
                      </div>
                      
                      <div className="flex flex-wrap gap-1 mt-2">
                        {profile.tags.slice(0, 3).map((tag) => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                        {profile.tags.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{profile.tags.length - 3}
                          </Badge>
                        )}
                      </div>
                      
                      <div className="flex space-x-1 mt-3">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => loadProfileForEdit(profile)}
                        >
                          <Edit className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            const newProfile = { ...profile, profile_name: `${profile.profile_name}_copy` };
                            loadProfileForEdit(newProfile);
                          }}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => onDeleteProfile(profile.id)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Configuration Edit Dialog */}
      <Dialog open={editingConfig} onOpenChange={setEditingConfig}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {selectedConfig ? 'Edit Configuration' : 'New Configuration'}
            </DialogTitle>
            <DialogDescription>
              Configure service settings and parameters
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {validationErrors.length > 0 && (
              <div className="p-3 bg-red-50 border border-red-200 rounded">
                <div className="flex items-center space-x-2 mb-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm font-medium text-red-700">Validation Errors</span>
                </div>
                <ul className="text-sm text-red-600 space-y-1">
                  {validationErrors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="service_name">Service Name</Label>
                <Input
                  id="service_name"
                  value={configForm.service_name}
                  onChange={(e) => setConfigForm({ ...configForm, service_name: e.target.value })}
                  placeholder="e.g., PostgreSQL"
                />
              </div>
              
              <div>
                <Label htmlFor="service_type">Service Type</Label>
                <Select
                  value={configForm.service_type}
                  onValueChange={(value) => setConfigForm({ ...configForm, service_type: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select service type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                    <SelectItem value="redis">Redis</SelectItem>
                    <SelectItem value="ollama">Ollama</SelectItem>
                    <SelectItem value="agent">Agent</SelectItem>
                    <SelectItem value="monitoring">Monitoring</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="environment">Environment</Label>
                <Select
                  value={configForm.environment}
                  onValueChange={(value) => setConfigForm({ ...configForm, environment: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="development">Development</SelectItem>
                    <SelectItem value="staging">Staging</SelectItem>
                    <SelectItem value="production">Production</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="version">Version</Label>
                <Input
                  id="version"
                  value={configForm.version}
                  onChange={(e) => setConfigForm({ ...configForm, version: e.target.value })}
                  placeholder="1.0.0"
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="configuration">Configuration (JSON)</Label>
              <Textarea
                id="configuration"
                value={configForm.configuration}
                onChange={(e) => setConfigForm({ ...configForm, configuration: e.target.value })}
                placeholder='{"port": 5432, "database": "pygent"}'
                rows={8}
                className="font-mono text-sm"
              />
            </div>
            
            <div className="flex space-x-2">
              <Button onClick={handleSaveConfiguration} className="flex-1">
                <Save className="h-4 w-4 mr-2" />
                Save Configuration
              </Button>
              <Button variant="outline" onClick={() => setEditingConfig(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Profile Edit Dialog */}
      <Dialog open={editingProfile} onOpenChange={setEditingProfile}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {selectedProfile ? 'Edit Profile' : 'New Profile'}
            </DialogTitle>
            <DialogDescription>
              Configure startup profile and service sequence
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {validationErrors.length > 0 && (
              <div className="p-3 bg-red-50 border border-red-200 rounded">
                <div className="flex items-center space-x-2 mb-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm font-medium text-red-700">Validation Errors</span>
                </div>
                <ul className="text-sm text-red-600 space-y-1">
                  {validationErrors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}
            
            <div>
              <Label htmlFor="profile_name">Profile Name</Label>
              <Input
                id="profile_name"
                value={profileForm.profile_name}
                onChange={(e) => setProfileForm({ ...profileForm, profile_name: e.target.value })}
                placeholder="e.g., Development Setup"
              />
            </div>
            
            <div>
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={profileForm.description}
                onChange={(e) => setProfileForm({ ...profileForm, description: e.target.value })}
                placeholder="Brief description of this profile"
                rows={2}
              />
            </div>
            
            <div>
              <Label htmlFor="startup_sequence">Startup Sequence (comma-separated)</Label>
              <Input
                id="startup_sequence"
                value={profileForm.startup_sequence.join(', ')}
                onChange={(e) => setProfileForm({ 
                  ...profileForm, 
                  startup_sequence: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                })}
                placeholder="PostgreSQL, Redis, Ollama"
              />
            </div>
            
            <div>
              <Label htmlFor="environment_variables">Environment Variables (JSON)</Label>
              <Textarea
                id="environment_variables"
                value={profileForm.environment_variables}
                onChange={(e) => setProfileForm({ ...profileForm, environment_variables: e.target.value })}
                placeholder='{"NODE_ENV": "development", "DEBUG": "true"}'
                rows={4}
                className="font-mono text-sm"
              />
            </div>
            
            <div>
              <Label htmlFor="tags">Tags (comma-separated)</Label>
              <Input
                id="tags"
                value={profileForm.tags.join(', ')}
                onChange={(e) => setProfileForm({ 
                  ...profileForm, 
                  tags: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                })}
                placeholder="development, quick-start, minimal"
              />
            </div>
            
            <div className="flex space-x-2">
              <Button onClick={handleSaveProfile} className="flex-1">
                <Save className="h-4 w-4 mr-2" />
                Save Profile
              </Button>
              <Button variant="outline" onClick={() => setEditingProfile(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfigurationManager;
