/**
 * Log Viewer Component
 * Advanced log streaming and filtering interface
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  FileText, 
  Search, 
  Filter,
  Download,
  Trash2,
  Pause,
  Play,
  RotateCcw,
  Info,
  AlertTriangle,
  XCircle,
  CheckCircle,
  Clock,
  ArrowDown
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { LogViewerProps, LogFilter } from './types';
import { StartupLogEvent } from '@/types';

const LogViewer: React.FC<LogViewerProps> = ({
  logs,
  onClear,
  onFilter,
  onExport,
  maxLines = 1000,
  autoScroll = true,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [serviceFilter, setServiceFilter] = useState('all');
  const [isPaused, setIsPaused] = useState(false);
  const [showDetails, setShowDetails] = useState<number | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && !isPaused && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll, isPaused]);

  const getLogIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info':
        return <Info className="h-4 w-4 text-blue-500" />;
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLogColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error':
        return 'border-l-red-500 bg-red-50 hover:bg-red-100';
      case 'warning':
        return 'border-l-yellow-500 bg-yellow-50 hover:bg-yellow-100';
      case 'info':
        return 'border-l-blue-500 bg-blue-50 hover:bg-blue-100';
      case 'success':
        return 'border-l-green-500 bg-green-50 hover:bg-green-100';
      default:
        return 'border-l-gray-500 bg-gray-50 hover:bg-gray-100';
    }
  };

  const filteredLogs = logs
    .filter(log => {
      // Level filter
      if (levelFilter !== 'all' && log.data.level !== levelFilter) {
        return false;
      }
      
      // Service filter
      if (serviceFilter !== 'all' && log.data.service !== serviceFilter) {
        return false;
      }
      
      // Search term filter
      if (searchTerm && !log.data.message.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      return true;
    })
    .slice(-maxLines); // Keep only the last N logs

  const uniqueServices = Array.from(new Set(logs.map(log => log.data.service).filter(Boolean)));
  const logLevelCounts = logs.reduce((acc, log) => {
    acc[log.data.level] = (acc[log.data.level] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const handleFilter = () => {
    const filter: LogFilter = {
      level: levelFilter !== 'all' ? [levelFilter] : undefined,
      service: serviceFilter !== 'all' ? [serviceFilter] : undefined,
      searchTerm: searchTerm || undefined
    };
    onFilter(filter);
  };

  const scrollToBottom = () => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Log Controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-blue-500" />
                <span>Log Viewer</span>
                <Badge variant="outline">{filteredLogs.length} entries</Badge>
              </CardTitle>
              <CardDescription>
                Real-time log streaming with filtering and search
              </CardDescription>
            </div>
            
            <div className="flex space-x-2">
              <Button
                size="sm"
                variant={isPaused ? "default" : "outline"}
                onClick={() => setIsPaused(!isPaused)}
              >
                {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
              </Button>
              
              <Button size="sm" variant="outline" onClick={scrollToBottom}>
                <ArrowDown className="h-4 w-4" />
              </Button>
              
              <Button size="sm" variant="outline" onClick={onExport}>
                <Download className="h-4 w-4" />
              </Button>
              
              <Button size="sm" variant="outline" onClick={onClear}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Filter Controls */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Input
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            
            <div>
              <Select value={levelFilter} onValueChange={setLevelFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="error">Error ({logLevelCounts.error || 0})</SelectItem>
                  <SelectItem value="warning">Warning ({logLevelCounts.warning || 0})</SelectItem>
                  <SelectItem value="info">Info ({logLevelCounts.info || 0})</SelectItem>
                  <SelectItem value="success">Success ({logLevelCounts.success || 0})</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Select value={serviceFilter} onValueChange={setServiceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by service" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Services</SelectItem>
                  {uniqueServices.map((service) => (
                    <SelectItem key={service} value={service}>
                      {service}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Button onClick={handleFilter} className="w-full">
                <Filter className="h-4 w-4 mr-2" />
                Apply Filters
              </Button>
            </div>
          </div>

          {/* Log Statistics */}
          <div className="flex space-x-4 text-sm text-muted-foreground">
            <span>Total: {logs.length}</span>
            <span>Filtered: {filteredLogs.length}</span>
            <span>Errors: {logLevelCounts.error || 0}</span>
            <span>Warnings: {logLevelCounts.warning || 0}</span>
            {isPaused && (
              <Badge variant="secondary" className="text-xs">
                <Pause className="h-3 w-3 mr-1" />
                Paused
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Log Display */}
      <Card>
        <CardContent className="p-0">
          <div 
            ref={logContainerRef}
            className="max-h-96 overflow-auto bg-gray-50 font-mono text-sm"
          >
            {filteredLogs.length > 0 ? (
              <div className="space-y-1 p-4">
                {filteredLogs.map((log, index) => (
                  <div
                    key={index}
                    className={cn(
                      "p-3 border-l-4 rounded-r cursor-pointer transition-colors",
                      getLogColor(log.data.level),
                      showDetails === index && "ring-2 ring-blue-500"
                    )}
                    onClick={() => setShowDetails(showDetails === index ? null : index)}
                  >
                    <div className="flex items-start space-x-3">
                      {getLogIcon(log.data.level)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline" className="text-xs">
                              {log.data.level.toUpperCase()}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {formatTimestamp(log.timestamp)}
                            </span>
                            {log.data.service && (
                              <Badge variant="secondary" className="text-xs">
                                {log.data.service}
                              </Badge>
                            )}
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {log.data.logger}
                          </span>
                        </div>
                        
                        <p className="text-sm mt-1 break-words">{log.data.message}</p>
                        
                        {/* Expanded Details */}
                        {showDetails === index && Object.keys(log.data.details).length > 0 && (
                          <div className="mt-3 p-2 bg-white rounded border">
                            <h5 className="text-xs font-semibold mb-2">Details:</h5>
                            <pre className="text-xs overflow-auto max-h-32">
                              {JSON.stringify(log.data.details, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No logs to display</p>
                <p className="text-sm">
                  {logs.length === 0 
                    ? 'No logs have been generated yet'
                    : 'No logs match the current filters'
                  }
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LogViewer;
