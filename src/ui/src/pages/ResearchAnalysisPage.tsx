import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Progress } from '../components/ui/progress';
import { Badge } from '../components/ui/badge';
import { Separator } from '../components/ui/separator';
import { 
  Search, 
  Brain, 
  Download, 
  FileText, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  BookOpen,
  BarChart3,
  ExternalLink
} from 'lucide-react';

interface WorkflowProgress {
  workflow_id: string;
  status: string;
  current_step: string;
  progress_percentage: number;
  research_papers_found: number;
  analysis_confidence: number;
  error_message?: string;
}

interface Citation {
  title: string;
  authors: string[];
  year: string;
  journal: string;
  url: string;
  doi?: string;
}

interface WorkflowResult {
  workflow_id: string;
  success: boolean;
  query: string;
  research_summary: string;
  analysis_summary: string;
  formatted_output: string;
  citations: Citation[];
  metadata: any;
  execution_time: number;
  error_message?: string;
}

const ResearchAnalysisPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [analysisModel, setAnalysisModel] = useState('deepseek-r1:8b');
  const [maxPapers, setMaxPapers] = useState(15);
  const [analysisDepth, setAnalysisDepth] = useState(3);
  
  const [isRunning, setIsRunning] = useState(false);
  const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(null);
  const [progress, setProgress] = useState<WorkflowProgress | null>(null);
  const [result, setResult] = useState<WorkflowResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Available models for analysis
  const availableModels = [
    'deepseek-r1:8b',
    'qwen3:8b', 
    'deepseek-coder:6.7b',
    'llama3.1:8b'
  ];

  const startWorkflow = async () => {
    if (!query.trim()) {
      setError('Please enter a research query');
      return;
    }

    setIsRunning(true);
    setError(null);
    setResult(null);
    setProgress(null);

    try {
      const response = await fetch('/api/v1/workflows/research-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          analysis_model: analysisModel,
          max_papers: maxPapers,
          analysis_depth: analysisDepth,
          export_format: 'markdown'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setCurrentWorkflowId(data.workflow_id);
      
      // Start polling for progress
      pollProgress(data.workflow_id);
      
    } catch (err) {
      setError(`Failed to start workflow: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setIsRunning(false);
    }
  };

  const pollProgress = async (workflowId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/workflows/research-analysis/${workflowId}/status`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const progressData: WorkflowProgress = await response.json();
        setProgress(progressData);

        // Check if workflow is complete
        if (progressData.status === 'completed') {
          clearInterval(pollInterval);
          await fetchResult(workflowId);
          setIsRunning(false);
        } else if (progressData.status === 'failed') {
          clearInterval(pollInterval);
          setError(progressData.error_message || 'Workflow failed');
          setIsRunning(false);
        }
        
      } catch (err) {
        clearInterval(pollInterval);
        setError(`Failed to get progress: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setIsRunning(false);
      }
    }, 1000);
  };

  const fetchResult = async (workflowId: string) => {
    try {
      const response = await fetch(`/api/v1/workflows/research-analysis/${workflowId}/result`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const resultData: WorkflowResult = await response.json();
      setResult(resultData);
      
    } catch (err) {
      setError(`Failed to get result: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const exportResult = async (format: string) => {
    if (!currentWorkflowId) return;

    try {
      const response = await fetch(`/api/v1/workflows/research-analysis/${currentWorkflowId}/export/${format}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `research_analysis_${currentWorkflowId}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      setError(`Failed to export: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'research_phase':
        return <Search className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'analysis_phase':
        return <Brain className="h-5 w-5 text-purple-500 animate-pulse" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Research-to-Analysis Workflow
        </h1>
        <p className="text-gray-600">
          Automated pipeline that searches academic databases and provides AI-powered analysis
        </p>
      </div>

      {/* Query Input Section */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Research Query
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Research Question</label>
            <Textarea
              placeholder="Enter your research query (e.g., 'quantum computing feasibility using larger qubits on silicon')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="min-h-[100px]"
              disabled={isRunning}
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Analysis Model</label>
              <Select value={analysisModel} onValueChange={setAnalysisModel} disabled={isRunning}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Max Papers</label>
              <Input
                type="number"
                min="5"
                max="50"
                value={maxPapers}
                onChange={(e) => setMaxPapers(parseInt(e.target.value))}
                disabled={isRunning}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Analysis Depth</label>
              <Select value={analysisDepth.toString()} onValueChange={(v) => setAnalysisDepth(parseInt(v))} disabled={isRunning}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Shallow (Fast)</SelectItem>
                  <SelectItem value="2">Medium</SelectItem>
                  <SelectItem value="3">Deep (Recommended)</SelectItem>
                  <SelectItem value="4">Very Deep</SelectItem>
                  <SelectItem value="5">Exhaustive</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <Button 
            onClick={startWorkflow} 
            disabled={isRunning || !query.trim()}
            className="w-full md:w-auto"
            size="lg"
          >
            {isRunning ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Running Workflow...
              </>
            ) : (
              <>
                <Search className="h-4 w-4 mr-2" />
                Research & Analyze
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-red-700">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">Error:</span>
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Progress Display */}
      {progress && isRunning && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(progress.status)}
              Workflow Progress
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>{progress.current_step}</span>
                <span>{Math.round(progress.progress_percentage)}%</span>
              </div>
              <Progress value={progress.progress_percentage} className="h-2" />
            </div>
            
            <div className="flex gap-4 text-sm">
              {progress.research_papers_found > 0 && (
                <Badge variant="secondary">
                  <BookOpen className="h-3 w-3 mr-1" />
                  {progress.research_papers_found} papers found
                </Badge>
              )}
              {progress.analysis_confidence > 0 && (
                <Badge variant="secondary">
                  <BarChart3 className="h-3 w-3 mr-1" />
                  {Math.round(progress.analysis_confidence * 100)}% confidence
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Display */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Research Data Panel */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5 text-blue-500" />
                Research Data
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <div className="bg-gray-50 p-4 rounded-lg mb-4">
                  <h4 className="font-medium mb-2">Research Summary</h4>
                  <p className="text-sm text-gray-700">{result.research_summary}</p>
                </div>
                
                {result.citations.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Key Sources ({result.citations.length})</h4>
                    <div className="space-y-2">
                      {result.citations.slice(0, 5).map((citation, index) => (
                        <div key={index} className="text-xs border-l-2 border-blue-200 pl-3">
                          <div className="font-medium">{citation.title}</div>
                          <div className="text-gray-600">
                            {citation.authors.join(', ')} ({citation.year})
                          </div>
                          {citation.url && (
                            <a 
                              href={citation.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline flex items-center gap-1"
                            >
                              <ExternalLink className="h-3 w-3" />
                              View Paper
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Analysis Panel */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-500" />
                AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <div className="bg-purple-50 p-4 rounded-lg mb-4">
                  <h4 className="font-medium mb-2">Analysis Summary</h4>
                  <p className="text-sm text-gray-700">{result.analysis_summary}</p>
                </div>
                
                <div className="flex gap-2 mb-4">
                  <Badge variant="outline">
                    Model: {result.metadata?.analysis_model || 'Unknown'}
                  </Badge>
                  <Badge variant="outline">
                    Confidence: {Math.round((result.metadata?.analysis_confidence || 0) * 100)}%
                  </Badge>
                  <Badge variant="outline">
                    Time: {result.execution_time?.toFixed(1)}s
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Full Report and Export */}
      {result && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Complete Report
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <div className="flex gap-2 mb-4">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => exportResult('markdown')}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export Markdown
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => exportResult('html')}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export HTML
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => exportResult('json')}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export JSON
                </Button>
              </div>
            </div>
            
            <Separator className="mb-4" />
            
            <div className="bg-gray-50 p-4 rounded-lg max-h-96 overflow-y-auto">
              <pre className="whitespace-pre-wrap text-sm font-serif leading-relaxed">
                {result.formatted_output}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ResearchAnalysisPage;
