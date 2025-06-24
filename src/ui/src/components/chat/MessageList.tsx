import React from 'react';
import { Bot, User, Brain, Dna, Search, Clock, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { ChatMessage, ReasoningContent, EvolutionContent } from '@/types';
import { cn } from '@/utils/cn';
import { format, parseISO } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MessageListProps {
  messages: ChatMessage[];
  typingUsers: Set<string>;
  className?: string;
}

// Helper function to safely parse timestamps
const parseTimestamp = (timestamp: Date | string): Date => {
  if (timestamp instanceof Date) {
    return timestamp;
  }
  if (typeof timestamp === 'string') {
    try {
      return parseISO(timestamp);
    } catch (error) {
      console.warn('Failed to parse timestamp:', timestamp, error);
      return new Date();
    }
  }
  return new Date();
};

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  typingUsers,
  className
}) => {
  const getMessageIcon = (message: ChatMessage) => {
    if (message.type === 'user') {
      return <User className="h-4 w-4" />;
    }

    switch (message.type) {
      case 'reasoning':
        return <Brain className="h-4 w-4 text-blue-500" />;
      case 'evolution':
        return <Dna className="h-4 w-4 text-green-500" />;
      case 'system':
        return <Search className="h-4 w-4 text-purple-500" />;
      default:
        return <Bot className="h-4 w-4 text-gray-500" />;
    }
  };

  const getMessageSender = (message: ChatMessage) => {
    if (message.type === 'user') return 'You';
    
    switch (message.type) {
      case 'reasoning':
        return 'Reasoning Agent';
      case 'evolution':
        return 'Evolution Agent';
      case 'system':
        return 'Search Agent';
      default:
        return 'AI Assistant';
    }
  };

  const renderMessageContent = (message: ChatMessage) => {
    if (typeof message.content === 'string') {
      return (
        <ReactMarkdown
          className="prose prose-sm max-w-none dark:prose-invert"
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {message.content}
        </ReactMarkdown>
      );
    }

    if (message.type === 'reasoning' && typeof message.content === 'object') {
      return <ReasoningMessageContent content={message.content as ReasoningContent} />;
    }

    if (message.type === 'evolution' && typeof message.content === 'object') {
      return <EvolutionMessageContent content={message.content as EvolutionContent} />;
    }

    return <div>Unsupported message type</div>;
  };

  const renderMetadata = (message: ChatMessage) => {
    if (!message.metadata) return null;

    return (
      <div className="flex items-center space-x-4 text-xs text-muted-foreground mt-2">
        {message.metadata.processing_time && (
          <div className="flex items-center space-x-1">
            <Clock className="h-3 w-3" />
            <span>{message.metadata.processing_time.toFixed(2)}s</span>
          </div>
        )}
        {message.metadata.confidence_score && (
          <div className="flex items-center space-x-1">
            <CheckCircle className="h-3 w-3" />
            <span>{(message.metadata.confidence_score * 100).toFixed(1)}%</span>
          </div>
        )}
        {message.metadata.documents_retrieved && (
          <div className="flex items-center space-x-1">
            <Search className="h-3 w-3" />
            <span>{message.metadata.documents_retrieved} docs</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={cn('flex flex-col space-y-4 p-4 overflow-y-auto', className)}>
      {messages.map((message) => (
        <div
          key={message.id}
          className={cn(
            'flex',
            message.type === 'user' ? 'justify-end' : 'justify-start'
          )}
        >
          <Card className={cn(
            'max-w-[80%]',
            message.type === 'user' 
              ? 'bg-primary text-primary-foreground' 
              : 'bg-muted'
          )}>
            <CardContent className="p-3">
              <div className="flex items-start space-x-2">
                <div className="flex-shrink-0 mt-1">
                  {getMessageIcon(message)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">
                      {getMessageSender(message)}
                    </span>
                    <span className="text-xs opacity-70">
                      {format(parseTimestamp(message.timestamp), 'HH:mm')}
                    </span>
                  </div>
                  <div className="text-sm">
                    {renderMessageContent(message)}
                  </div>
                  {renderMetadata(message)}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      ))}
      
      {typingUsers.size > 0 && (
        <div className="flex justify-start">
          <Card className="bg-muted">
            <CardContent className="p-3">
              <div className="flex items-center space-x-2">
                <Bot className="h-4 w-4 text-gray-500" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

const ReasoningMessageContent: React.FC<{ content: ReasoningContent }> = ({ content }) => {
  return (
    <div className="space-y-3">
      <div>
        <h4 className="font-medium mb-1">Response:</h4>
        <ReactMarkdown className="prose prose-sm max-w-none dark:prose-invert">
          {content.response}
        </ReactMarkdown>
      </div>
      
      {content.reasoning_path && content.reasoning_path.length > 0 && (
        <div>
          <h4 className="font-medium mb-2">Reasoning Path:</h4>
          <div className="space-y-2">
            {content.reasoning_path.slice(0, 3).map((thought, index) => (
              <div key={thought.id} className="text-xs bg-background/50 rounded p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">Step {index + 1}</span>
                  <span className="text-muted-foreground">
                    Score: {thought.value_score.toFixed(2)}
                  </span>
                </div>
                <p>{thought.content}</p>
              </div>
            ))}
            {content.reasoning_path.length > 3 && (
              <p className="text-xs text-muted-foreground">
                +{content.reasoning_path.length - 3} more reasoning steps...
              </p>
            )}
          </div>
        </div>
      )}
      
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Mode: {content.task_complexity}</span>
        <span>Confidence: {(content.confidence_score * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
};

const EvolutionMessageContent: React.FC<{ content: EvolutionContent }> = ({ content }) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-medium">Generation {content.generation}</h4>
        <span className="text-sm text-muted-foreground">
          Fitness: {content.fitness_score.toFixed(3)}
        </span>
      </div>
      
      {content.recipe_changes && content.recipe_changes.length > 0 && (
        <div>
          <h5 className="text-sm font-medium mb-1">Recipe Changes:</h5>
          <ul className="text-xs space-y-1">
            {content.recipe_changes.map((change, index) => (
              <li key={index} className="flex items-start space-x-1">
                <span className="text-green-500">•</span>
                <span>{change}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-muted-foreground">Population:</span>
          <span className="ml-1">{content.population_size}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Status:</span>
          <span className="ml-1">{content.convergence_status}</span>
        </div>
      </div>
    </div>
  );
};
