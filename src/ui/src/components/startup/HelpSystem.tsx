/**
 * Help System Component
 * Contextual help and guided tours for startup service interface
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { 
  HelpCircle, 
  Book, 
  Play,
  ChevronDown,
  ChevronRight,
  Search,
  ExternalLink,
  Lightbulb,
  Video,
  FileText,
  Zap,
  Settings,
  Users,
  BarChart3,
  GitBranch
} from 'lucide-react';
import { cn } from '@/utils/cn';

interface HelpTopic {
  id: string;
  title: string;
  description: string;
  category: 'getting-started' | 'configuration' | 'orchestration' | 'monitoring' | 'troubleshooting';
  content: string;
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface GuidedTour {
  id: string;
  title: string;
  description: string;
  steps: {
    target: string;
    title: string;
    content: string;
    position: 'top' | 'bottom' | 'left' | 'right';
  }[];
}

interface HelpSystemProps {
  currentPage?: string;
  className?: string;
}

const HelpSystem: React.FC<HelpSystemProps> = ({
  currentPage = 'dashboard',
  className
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedTopics, setExpandedTopics] = useState<string[]>([]);
  const [activeTour, setActiveTour] = useState<string | null>(null);

  const helpTopics: HelpTopic[] = [
    {
      id: 'getting-started',
      title: 'Getting Started with Startup Service',
      description: 'Learn the basics of PyGent Factory startup service management',
      category: 'getting-started',
      difficulty: 'beginner',
      tags: ['basics', 'introduction', 'setup'],
      content: `
# Getting Started

The PyGent Factory Startup Service provides comprehensive orchestration and management capabilities for your development environment.

## Key Features
- **Service Orchestration**: Manage startup sequences and dependencies
- **Configuration Management**: Create and manage service configurations
- **Real-time Monitoring**: Monitor service health and performance
- **Profile Management**: Save and reuse startup configurations

## Quick Start
1. Navigate to the Orchestration page
2. Select or create a startup profile
3. Configure your services and dependencies
4. Start your services with one click

## Next Steps
- Explore the Configuration page to customize service settings
- Use the Monitoring page to track service performance
- Create custom profiles for different environments
      `
    },
    {
      id: 'service-configuration',
      title: 'Service Configuration Guide',
      description: 'How to configure individual services and their settings',
      category: 'configuration',
      difficulty: 'intermediate',
      tags: ['configuration', 'services', 'settings'],
      content: `
# Service Configuration

Configure individual services to match your development needs.

## Configuration Types
- **PostgreSQL**: Database connection settings, port configuration
- **Redis**: Cache settings, memory limits, persistence options
- **Ollama**: AI model configuration, GPU settings
- **Custom Services**: Environment variables, startup commands

## Best Practices
- Use environment-specific configurations
- Document configuration changes
- Test configurations before deployment
- Use templates for common setups

## Configuration Validation
The system automatically validates configurations and provides feedback on potential issues.
      `
    },
    {
      id: 'orchestration-basics',
      title: 'Service Orchestration Fundamentals',
      description: 'Understanding service dependencies and startup sequences',
      category: 'orchestration',
      difficulty: 'intermediate',
      tags: ['orchestration', 'dependencies', 'sequences'],
      content: `
# Service Orchestration

Orchestrate complex service startup sequences with dependency management.

## Dependency Management
- Define service dependencies
- Automatic dependency resolution
- Circular dependency detection
- Parallel execution support

## Execution Modes
- **Sequential**: Start services one after another
- **Parallel**: Start independent services simultaneously
- **Mixed**: Combine sequential and parallel execution

## Monitoring Execution
- Real-time progress tracking
- Service health monitoring
- Error detection and recovery
- Performance metrics
      `
    },
    {
      id: 'monitoring-setup',
      title: 'Monitoring and Logging Setup',
      description: 'Configure monitoring, logging, and alerting for your services',
      category: 'monitoring',
      difficulty: 'intermediate',
      tags: ['monitoring', 'logging', 'alerts'],
      content: `
# Monitoring Setup

Set up comprehensive monitoring for your startup services.

## Log Management
- Real-time log streaming
- Log filtering and search
- Log level configuration
- Export and archiving

## Performance Metrics
- CPU and memory usage
- Service response times
- Health check status
- Custom metrics

## Alerting
- Configure alert thresholds
- Email and webhook notifications
- Escalation policies
- Alert history
      `
    },
    {
      id: 'troubleshooting',
      title: 'Common Issues and Troubleshooting',
      description: 'Resolve common startup service issues and errors',
      category: 'troubleshooting',
      difficulty: 'beginner',
      tags: ['troubleshooting', 'errors', 'debugging'],
      content: `
# Troubleshooting Guide

Common issues and their solutions.

## Service Won't Start
1. Check service configuration
2. Verify dependencies are running
3. Review error logs
4. Check port availability
5. Validate environment variables

## Performance Issues
- Monitor resource usage
- Check for memory leaks
- Optimize configuration
- Review dependency chains

## Connection Problems
- Verify network connectivity
- Check firewall settings
- Validate service endpoints
- Test authentication

## Getting Help
- Check the logs for detailed error messages
- Use the integration testing dashboard
- Contact support with specific error details
      `
    }
  ];

  const guidedTours: GuidedTour[] = [
    {
      id: 'dashboard-tour',
      title: 'Dashboard Overview',
      description: 'Get familiar with the startup service dashboard',
      steps: [
        {
          target: '.service-status-card',
          title: 'Service Status',
          content: 'Monitor the current status of all your services here.',
          position: 'bottom'
        },
        {
          target: '.orchestration-controls',
          title: 'Orchestration Controls',
          content: 'Start, stop, and manage service sequences from this panel.',
          position: 'left'
        },
        {
          target: '.monitoring-panel',
          title: 'Monitoring Panel',
          content: 'View real-time metrics and logs for your services.',
          position: 'top'
        }
      ]
    },
    {
      id: 'configuration-tour',
      title: 'Configuration Walkthrough',
      description: 'Learn how to configure services and profiles',
      steps: [
        {
          target: '.configuration-tabs',
          title: 'Configuration Sections',
          content: 'Switch between service configurations and startup profiles.',
          position: 'bottom'
        },
        {
          target: '.service-config-form',
          title: 'Service Configuration',
          content: 'Configure individual service settings and parameters.',
          position: 'right'
        },
        {
          target: '.profile-manager',
          title: 'Profile Management',
          content: 'Create and manage startup profiles for different environments.',
          position: 'left'
        }
      ]
    }
  ];

  const filteredTopics = helpTopics.filter(topic => {
    const matchesSearch = !searchTerm || 
      topic.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      topic.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      topic.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || topic.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  const toggleTopic = (topicId: string) => {
    setExpandedTopics(prev => 
      prev.includes(topicId) 
        ? prev.filter(id => id !== topicId)
        : [...prev, topicId]
    );
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'getting-started':
        return <Play className="h-4 w-4 text-green-500" />;
      case 'configuration':
        return <Settings className="h-4 w-4 text-blue-500" />;
      case 'orchestration':
        return <GitBranch className="h-4 w-4 text-purple-500" />;
      case 'monitoring':
        return <BarChart3 className="h-4 w-4 text-orange-500" />;
      case 'troubleshooting':
        return <Lightbulb className="h-4 w-4 text-yellow-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner':
        return 'bg-green-100 text-green-800';
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800';
      case 'advanced':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Help System Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <HelpCircle className="h-5 w-5 text-blue-500" />
                <span>Help & Documentation</span>
                <Badge variant="outline">{filteredTopics.length} topics</Badge>
              </CardTitle>
              <CardDescription>
                Get help, learn features, and troubleshoot issues
              </CardDescription>
            </div>
            
            <div className="flex space-x-2">
              <Button variant="outline">
                <Video className="h-4 w-4 mr-2" />
                Video Tutorials
              </Button>
              <Button variant="outline">
                <ExternalLink className="h-4 w-4 mr-2" />
                Documentation
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Search and Filters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search help topics..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            
            <div>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
              >
                <option value="all">All Categories</option>
                <option value="getting-started">Getting Started</option>
                <option value="configuration">Configuration</option>
                <option value="orchestration">Orchestration</option>
                <option value="monitoring">Monitoring</option>
                <option value="troubleshooting">Troubleshooting</option>
              </select>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline">
              <Play className="h-3 w-3 mr-1" />
              Quick Start Guide
            </Button>
            <Button size="sm" variant="outline">
              <Zap className="h-3 w-3 mr-1" />
              Common Tasks
            </Button>
            <Button size="sm" variant="outline">
              <Lightbulb className="h-3 w-3 mr-1" />
              Troubleshooting
            </Button>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="help-topics" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="help-topics">Help Topics</TabsTrigger>
          <TabsTrigger value="guided-tours">Guided Tours</TabsTrigger>
          <TabsTrigger value="quick-reference">Quick Reference</TabsTrigger>
        </TabsList>

        {/* Help Topics Tab */}
        <TabsContent value="help-topics" className="space-y-4">
          <div className="space-y-3">
            {filteredTopics.map((topic) => (
              <Card key={topic.id}>
                <Collapsible
                  open={expandedTopics.includes(topic.id)}
                  onOpenChange={() => toggleTopic(topic.id)}
                >
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {getCategoryIcon(topic.category)}
                          <div>
                            <CardTitle className="text-lg">{topic.title}</CardTitle>
                            <CardDescription>{topic.description}</CardDescription>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className={getDifficultyColor(topic.difficulty)}>
                            {topic.difficulty}
                          </Badge>
                          {expandedTopics.includes(topic.id) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-1 mt-2">
                        {topic.tags.map((tag) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </CardHeader>
                  </CollapsibleTrigger>
                  
                  <CollapsibleContent>
                    <CardContent>
                      <div className="prose prose-sm max-w-none">
                        <pre className="whitespace-pre-wrap text-sm">
                          {topic.content}
                        </pre>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Guided Tours Tab */}
        <TabsContent value="guided-tours" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {guidedTours.map((tour) => (
              <Card key={tour.id}>
                <CardContent className="p-6">
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-blue-100 rounded">
                      <Play className="h-5 w-5 text-blue-600" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold mb-2">{tour.title}</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        {tour.description}
                      </p>
                      
                      <div className="space-y-2 mb-4">
                        <p className="text-xs font-medium">Tour includes:</p>
                        <ul className="text-xs text-muted-foreground space-y-1">
                          {tour.steps.map((step, index) => (
                            <li key={index}>• {step.title}</li>
                          ))}
                        </ul>
                      </div>
                      
                      <Button size="sm" className="w-full">
                        <Play className="h-3 w-3 mr-1" />
                        Start Tour
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Quick Reference Tab */}
        <TabsContent value="quick-reference" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Keyboard Shortcuts</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Start Services</span>
                  <Badge variant="outline">Ctrl + S</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Stop Services</span>
                  <Badge variant="outline">Ctrl + X</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Refresh Status</span>
                  <Badge variant="outline">F5</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Open Help</span>
                  <Badge variant="outline">F1</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Service Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Running - Service is active</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Starting - Service is initializing</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                  <span>Stopped - Service is inactive</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Error - Service has failed</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Common Commands</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  <p className="font-medium">Start Profile</p>
                  <p className="text-muted-foreground">Select profile → Start sequence</p>
                </div>
                <div>
                  <p className="font-medium">View Logs</p>
                  <p className="text-muted-foreground">Monitoring → System Logs</p>
                </div>
                <div>
                  <p className="font-medium">Create Configuration</p>
                  <p className="text-muted-foreground">Configuration → New Configuration</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default HelpSystem;
