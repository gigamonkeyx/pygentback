import React from 'react';
import { Brain, Dna, Search, Bot, Code, BookOpen, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { AgentType } from './ChatInterface';
import { cn } from '@/utils/cn';

interface AgentSelectorProps {
  value: AgentType;
  onChange: (agent: AgentType) => void;
  className?: string;
}

const agents: Array<{
  id: AgentType;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}> = [
  {
    id: 'reasoning',
    name: 'Reasoning Agent',
    description: 'Advanced Tree of Thought reasoning with multi-path exploration',
    icon: <Brain className="h-4 w-4" />,
    color: 'text-blue-500'
  },
  {
    id: 'evolution',
    name: 'Evolution Agent',
    description: 'AI-guided recipe evolution and optimization',
    icon: <Dna className="h-4 w-4" />,
    color: 'text-green-500'
  },
  {
    id: 'search',
    name: 'Search Agent',
    description: 'GPU-accelerated vector search and document retrieval',
    icon: <Search className="h-4 w-4" />,
    color: 'text-purple-500'
  },
  {
    id: 'general',
    name: 'General Agent',
    description: 'General AI assistant for various tasks',
    icon: <Bot className="h-4 w-4" />,
    color: 'text-gray-500'
  },
  {
    id: 'coding',
    name: 'Coding Agent',
    description: 'Expert code generation, analysis, debugging, and optimization',
    icon: <Code className="h-4 w-4" />,
    color: 'text-orange-500'
  },
  {
    id: 'research',
    name: 'Research Agent',
    description: 'Academic and historical research with literature review capabilities',
    icon: <BookOpen className="h-4 w-4" />,
    color: 'text-indigo-500'
  }
];

export const AgentSelector: React.FC<AgentSelectorProps> = ({
  value,
  onChange,
  className
}) => {
  const currentAgent = agents.find(agent => agent.id === value) || agents[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          className={cn('justify-between min-w-[200px]', className)}
        >
          <div className="flex items-center space-x-2">
            <span className={currentAgent.color}>
              {currentAgent.icon}
            </span>
            <span className="text-sm font-medium">
              {currentAgent.name}
            </span>
          </div>
          <ChevronDown className="h-4 w-4 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-80" align="end">
        {agents.map((agent) => (
          <DropdownMenuItem
            key={agent.id}
            onClick={() => onChange(agent.id)}
            className={cn(
              'flex flex-col items-start space-y-1 p-3 cursor-pointer',
              value === agent.id && 'bg-accent'
            )}
          >
            <div className="flex items-center space-x-2 w-full">
              <span className={agent.color}>
                {agent.icon}
              </span>
              <span className="font-medium">
                {agent.name}
              </span>
              {value === agent.id && (
                <div className="ml-auto h-2 w-2 rounded-full bg-primary" />
              )}
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {agent.description}
            </p>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
