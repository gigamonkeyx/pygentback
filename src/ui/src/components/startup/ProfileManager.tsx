/**
 * Profile Manager Component
 * Advanced startup profile management with templates and sharing
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { 
  FileText, 
  Plus, 
  Edit, 
  Copy,
  Share2,
  Download,
  Upload,
  Star,
  MoreVertical,
  Play,
  Settings,
  Users,
  Layout,
  Zap,
  CheckCircle,
  Clock
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ProfileManagerProps } from './types';
import { ConfigurationProfile } from '@/types';

const ProfileManager: React.FC<ProfileManagerProps> = ({
  profiles,
  selectedProfile,
  onSelectProfile,
  onCreateProfile,
  onUpdateProfile,
  onDeleteProfile,
  onDuplicateProfile,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterTag, setFilterTag] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [showTemplates, setShowTemplates] = useState(false);
  const [editingProfile, setEditingProfile] = useState<ConfigurationProfile | null>(null);

  // Profile templates
  const profileTemplates = [
    {
      id: 'minimal',
      name: 'Minimal Setup',
      description: 'Basic services for development',
      services: ['PostgreSQL', 'Redis'],
      tags: ['minimal', 'development'],
      icon: <Zap className="h-4 w-4" />
    },
    {
      id: 'full-stack',
      name: 'Full Stack Development',
      description: 'Complete development environment',
      services: ['PostgreSQL', 'Redis', 'Ollama', 'Agent Orchestrator'],
      tags: ['full-stack', 'development'],
      icon: <Settings className="h-4 w-4" />
    },
    {
      id: 'ai-focused',
      name: 'AI/ML Focused',
      description: 'AI and machine learning services',
      services: ['Ollama', 'Agent Orchestrator', 'Vector Database'],
      tags: ['ai', 'ml', 'development'],
      icon: <Zap className="h-4 w-4" />
    },
    {
      id: 'production',
      name: 'Production Ready',
      description: 'Production deployment configuration',
      services: ['PostgreSQL', 'Redis', 'Monitoring', 'Load Balancer'],
      tags: ['production', 'monitoring'],
      icon: <CheckCircle className="h-4 w-4" />
    }
  ];

  const filteredProfiles = profiles.filter(profile => {
    // Search filter
    if (searchTerm && !profile.profile_name.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    
    // Tag filter
    if (filterTag !== 'all' && !profile.tags.includes(filterTag)) {
      return false;
    }
    
    return true;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.profile_name.localeCompare(b.profile_name);
      case 'usage':
        return b.usage_count - a.usage_count;
      case 'recent':
        return new Date(b.last_used || 0).getTime() - new Date(a.last_used || 0).getTime();
      default:
        return 0;
    }
  });

  const allTags = Array.from(new Set(profiles.flatMap(p => p.tags)));

  const handleCreateFromTemplate = (template: any) => {
    const newProfile = {
      profile_name: template.name,
      description: template.description,
      profile_type: 'template',
      services_config: {},
      startup_sequence: template.services,
      environment_variables: {},
      is_default: false,
      is_active: false,
      tags: template.tags,
      usage_count: 0
    };
    
    onCreateProfile(newProfile);
    setShowTemplates(false);
  };

  const handleExportProfile = (profile: ConfigurationProfile) => {
    const exportData = {
      ...profile,
      exported_at: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${profile.profile_name.replace(/\s+/g, '_')}_profile.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getProfileIcon = (profile: ConfigurationProfile) => {
    if (profile.is_default) return <Star className="h-4 w-4 text-yellow-500" />;
    if (profile.profile_type === 'template') return <Layout className="h-4 w-4 text-blue-500" />;
    return <FileText className="h-4 w-4 text-gray-500" />;
  };

  const formatLastUsed = (date?: Date) => {
    if (!date) return 'Never';
    
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    return `${Math.floor(days / 30)} months ago`;
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Profile Management Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <Users className="h-5 w-5 text-blue-500" />
                <span>Profile Management</span>
                <Badge variant="outline">{profiles.length} profiles</Badge>
              </CardTitle>
              <CardDescription>
                Manage startup profiles, templates, and configurations
              </CardDescription>
            </div>
            
            <div className="flex space-x-2">
              <Button variant="outline" onClick={() => setShowTemplates(true)}>
                <Layout className="h-4 w-4 mr-2" />
                Templates
              </Button>
              <Button onClick={() => onCreateProfile({})}>
                <Plus className="h-4 w-4 mr-2" />
                New Profile
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Search and Filters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Input
                placeholder="Search profiles..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div>
              <Select value={filterTag} onValueChange={setFilterTag}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by tag" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Tags</SelectItem>
                  {allTags.map((tag) => (
                    <SelectItem key={tag} value={tag}>
                      {tag}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger>
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="usage">Usage Count</SelectItem>
                  <SelectItem value="recent">Recently Used</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Button variant="outline" className="w-full">
                <Upload className="h-4 w-4 mr-2" />
                Import
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Profiles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredProfiles.map((profile) => (
          <Card 
            key={profile.id}
            className={cn(
              "cursor-pointer transition-all hover:shadow-md",
              selectedProfile?.id === profile.id && "ring-2 ring-blue-500",
              profile.is_default && "border-yellow-500"
            )}
            onClick={() => onSelectProfile(profile)}
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center space-x-2">
                  {getProfileIcon(profile)}
                  <h4 className="font-semibold truncate">{profile.profile_name}</h4>
                </div>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>Profile Actions</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => setEditingProfile(profile)}>
                      <Edit className="h-4 w-4 mr-2" />
                      Edit
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onDuplicateProfile(profile)}>
                      <Copy className="h-4 w-4 mr-2" />
                      Duplicate
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExportProfile(profile)}>
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>
                      <Share2 className="h-4 w-4 mr-2" />
                      Share
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {profile.description || 'No description provided'}
                </p>
                
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{profile.startup_sequence.length} services</span>
                  <span>Used {profile.usage_count} times</span>
                </div>
                
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Last used: {formatLastUsed(profile.last_used)}</span>
                  {profile.is_active && (
                    <Badge variant="default" className="bg-green-500 text-xs">
                      Active
                    </Badge>
                  )}
                </div>
              </div>
              
              {/* Tags */}
              <div className="flex flex-wrap gap-1 mt-3">
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
              
              {/* Quick Actions */}
              <div className="flex space-x-2 mt-3">
                <Button size="sm" className="flex-1">
                  <Play className="h-3 w-3 mr-1" />
                  Start
                </Button>
                <Button size="sm" variant="outline">
                  <Settings className="h-3 w-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {filteredProfiles.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Users className="h-16 w-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-semibold mb-2">No Profiles Found</h3>
            <p className="text-gray-600 mb-4">
              {searchTerm || filterTag !== 'all' 
                ? 'No profiles match your current filters.'
                : 'Create your first startup profile to get started.'
              }
            </p>
            <div className="flex space-x-2 justify-center">
              <Button onClick={() => onCreateProfile({})}>
                <Plus className="h-4 w-4 mr-2" />
                Create Profile
              </Button>
              <Button variant="outline" onClick={() => setShowTemplates(true)}>
                <Layout className="h-4 w-4 mr-2" />
                Use Template
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Profile Templates Dialog */}
      <Dialog open={showTemplates} onOpenChange={setShowTemplates}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Profile Templates</DialogTitle>
            <DialogDescription>
              Choose from pre-configured templates to quickly create new profiles
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {profileTemplates.map((template) => (
              <Card key={template.id} className="cursor-pointer hover:shadow-md transition-all">
                <CardContent className="p-4">
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-blue-100 rounded">
                      {template.icon}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold">{template.name}</h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        {template.description}
                      </p>
                      
                      <div className="space-y-2">
                        <div>
                          <p className="text-xs font-medium">Services:</p>
                          <div className="flex flex-wrap gap-1">
                            {template.services.map((service) => (
                              <Badge key={service} variant="secondary" className="text-xs">
                                {service}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <p className="text-xs font-medium">Tags:</p>
                          <div className="flex flex-wrap gap-1">
                            {template.tags.map((tag) => (
                              <Badge key={tag} variant="outline" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <Button 
                        size="sm" 
                        className="w-full mt-3"
                        onClick={() => handleCreateFromTemplate(template)}
                      >
                        Use Template
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ProfileManager;
