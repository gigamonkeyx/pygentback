/**
 * Dependency Graph Component
 * Visual dependency graph for service orchestration
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  GitBranch, 
  Play, 
  Square, 
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ArrowRight,
  Maximize2,
  Minimize2,
  RotateCcw,
  Settings
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ServiceStatus, SequenceStatus } from '@/types';

interface DependencyNode {
  id: string;
  name: string;
  status: ServiceStatus;
  dependencies: string[];
  position: { x: number; y: number };
  level: number;
}

interface DependencyGraphProps {
  services: string[];
  dependencies: Record<string, string[]>;
  serviceStatuses?: Record<string, ServiceStatus>;
  onServiceClick?: (serviceName: string) => void;
  className?: string;
}

const DependencyGraph: React.FC<DependencyGraphProps> = ({
  services,
  dependencies,
  serviceStatuses = {},
  onServiceClick,
  className
}) => {
  const [nodes, setNodes] = useState<DependencyNode[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Calculate node positions based on dependencies
  useEffect(() => {
    const calculateLayout = () => {
      const nodeMap = new Map<string, DependencyNode>();
      const levels: string[][] = [];
      const visited = new Set<string>();
      const visiting = new Set<string>();

      // Topological sort to determine levels
      const visit = (serviceName: string, level: number = 0): number => {
        if (visiting.has(serviceName)) {
          // Circular dependency detected
          console.warn(`Circular dependency detected involving ${serviceName}`);
          return level;
        }
        
        if (visited.has(serviceName)) {
          return nodeMap.get(serviceName)?.level || level;
        }

        visiting.add(serviceName);
        
        const serviceDeps = dependencies[serviceName] || [];
        let maxDepLevel = level;
        
        for (const dep of serviceDeps) {
          if (services.includes(dep)) {
            const depLevel = visit(dep, level + 1);
            maxDepLevel = Math.max(maxDepLevel, depLevel);
          }
        }

        visiting.delete(serviceName);
        visited.add(serviceName);

        const nodeLevel = maxDepLevel;
        if (!levels[nodeLevel]) levels[nodeLevel] = [];
        levels[nodeLevel].push(serviceName);

        nodeMap.set(serviceName, {
          id: serviceName,
          name: serviceName,
          status: serviceStatuses[serviceName] || ServiceStatus.UNKNOWN,
          dependencies: serviceDeps,
          position: { x: 0, y: 0 },
          level: nodeLevel
        });

        return nodeLevel;
      };

      // Visit all services
      services.forEach(service => visit(service));

      // Calculate positions
      const nodeWidth = 120;
      const nodeHeight = 60;
      const levelSpacing = 150;
      const nodeSpacing = 20;

      levels.forEach((levelServices, levelIndex) => {
        const totalWidth = levelServices.length * nodeWidth + (levelServices.length - 1) * nodeSpacing;
        const startX = -totalWidth / 2;

        levelServices.forEach((serviceName, serviceIndex) => {
          const node = nodeMap.get(serviceName);
          if (node) {
            node.position = {
              x: startX + serviceIndex * (nodeWidth + nodeSpacing) + nodeWidth / 2,
              y: levelIndex * levelSpacing + nodeHeight / 2
            };
          }
        });
      });

      setNodes(Array.from(nodeMap.values()));
    };

    calculateLayout();
  }, [services, dependencies, serviceStatuses]);

  const getStatusColor = (status: ServiceStatus) => {
    switch (status) {
      case ServiceStatus.RUNNING:
        return 'fill-green-500 stroke-green-600';
      case ServiceStatus.STARTING:
        return 'fill-yellow-500 stroke-yellow-600';
      case ServiceStatus.STOPPING:
        return 'fill-orange-500 stroke-orange-600';
      case ServiceStatus.STOPPED:
        return 'fill-gray-500 stroke-gray-600';
      case ServiceStatus.ERROR:
        return 'fill-red-500 stroke-red-600';
      default:
        return 'fill-gray-300 stroke-gray-400';
    }
  };

  const getStatusIcon = (status: ServiceStatus) => {
    switch (status) {
      case ServiceStatus.RUNNING:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case ServiceStatus.STARTING:
        return <Clock className="h-4 w-4 text-yellow-500 animate-spin" />;
      case ServiceStatus.STOPPING:
        return <Clock className="h-4 w-4 text-orange-500" />;
      case ServiceStatus.STOPPED:
        return <Square className="h-4 w-4 text-gray-500" />;
      case ServiceStatus.ERROR:
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-400" />;
    }
  };

  const renderConnections = () => {
    const connections: JSX.Element[] = [];

    nodes.forEach(node => {
      node.dependencies.forEach(depName => {
        const depNode = nodes.find(n => n.name === depName);
        if (depNode) {
          const key = `${node.id}-${depNode.id}`;
          connections.push(
            <line
              key={key}
              x1={depNode.position.x}
              y1={depNode.position.y + 30}
              x2={node.position.x}
              y2={node.position.y - 30}
              stroke="#6b7280"
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
              className="transition-all duration-200"
            />
          );
        }
      });
    });

    return connections;
  };

  const viewBox = nodes.length > 0 
    ? `${Math.min(...nodes.map(n => n.position.x)) - 100} ${Math.min(...nodes.map(n => n.position.y)) - 50} ${Math.max(...nodes.map(n => n.position.x)) - Math.min(...nodes.map(n => n.position.x)) + 200} ${Math.max(...nodes.map(n => n.position.y)) - Math.min(...nodes.map(n => n.position.y)) + 100}`
    : "0 0 400 300";

  return (
    <div className={cn('space-y-4', className)}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <GitBranch className="h-5 w-5 text-blue-500" />
              <span>Service Dependencies</span>
              <Badge variant="outline">{nodes.length} services</Badge>
            </CardTitle>
            <CardDescription>
              Visual representation of service startup dependencies
            </CardDescription>
          </div>
          
          <div className="flex space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
            <Button size="sm" variant="outline">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className={cn(
            "border rounded-lg bg-gray-50 overflow-auto transition-all duration-300",
            isExpanded ? "h-96" : "h-64"
          )}>
            {nodes.length > 0 ? (
              <svg
                ref={svgRef}
                className="w-full h-full"
                viewBox={viewBox}
                preserveAspectRatio="xMidYMid meet"
              >
                {/* Arrow marker definition */}
                <defs>
                  <marker
                    id="arrowhead"
                    markerWidth="10"
                    markerHeight="7"
                    refX="9"
                    refY="3.5"
                    orient="auto"
                  >
                    <polygon
                      points="0 0, 10 3.5, 0 7"
                      fill="#6b7280"
                    />
                  </marker>
                </defs>
                
                {/* Render connections */}
                {renderConnections()}
                
                {/* Render nodes */}
                {nodes.map(node => (
                  <g key={node.id}>
                    {/* Node background */}
                    <rect
                      x={node.position.x - 60}
                      y={node.position.y - 30}
                      width="120"
                      height="60"
                      rx="8"
                      className={cn(
                        "transition-all duration-200 cursor-pointer",
                        getStatusColor(node.status),
                        selectedNode === node.id && "stroke-blue-500 stroke-2",
                        "hover:opacity-80"
                      )}
                      onClick={() => {
                        setSelectedNode(selectedNode === node.id ? null : node.id);
                        onServiceClick?.(node.name);
                      }}
                    />
                    
                    {/* Node text */}
                    <text
                      x={node.position.x}
                      y={node.position.y - 5}
                      textAnchor="middle"
                      className="fill-white text-sm font-medium pointer-events-none"
                    >
                      {node.name}
                    </text>
                    
                    {/* Status text */}
                    <text
                      x={node.position.x}
                      y={node.position.y + 10}
                      textAnchor="middle"
                      className="fill-white text-xs opacity-90 pointer-events-none"
                    >
                      {node.status}
                    </text>
                  </g>
                ))}
              </svg>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                <div className="text-center">
                  <GitBranch className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No services to display</p>
                  <p className="text-sm">Add services to see the dependency graph</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex flex-wrap gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>Running</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span>Starting</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-500 rounded"></div>
              <span>Stopped</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>Error</span>
            </div>
          </div>
          
          {/* Selected Node Details */}
          {selectedNode && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <div className="flex items-center space-x-2 mb-2">
                {getStatusIcon(nodes.find(n => n.id === selectedNode)?.status || ServiceStatus.UNKNOWN)}
                <h4 className="font-semibold">{selectedNode}</h4>
              </div>
              
              <div className="text-sm text-muted-foreground">
                <p>Status: {nodes.find(n => n.id === selectedNode)?.status}</p>
                <p>Dependencies: {nodes.find(n => n.id === selectedNode)?.dependencies.join(', ') || 'None'}</p>
                <p>Level: {nodes.find(n => n.id === selectedNode)?.level}</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default DependencyGraph;
