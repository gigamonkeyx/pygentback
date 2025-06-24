import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { websocketService } from '@/services/websocket';

export const ChatInterface: React.FC = () => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState('general');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { messages, addMessage } = useAppStore();

  const agents = [
    { id: 'general', name: 'General AI', description: 'General purpose assistance' },
    { id: 'tot_reasoning', name: 'ToT Reasoning', description: 'Tree of Thought reasoning' },
    { id: 'rag_retrieval', name: 'RAG Search', description: 'Knowledge retrieval and search' },
    { id: 'evolution', name: 'Evolution', description: 'Recipe optimization' }
  ];

  useEffect(() => {
    // Set up WebSocket event handlers for chat
    websocketService.on('chat_response', (data) => {
      addMessage({
        content: data.content,
        role: 'assistant',
        agent: data.agent || selectedAgent,
        metadata: data.metadata
      });
      setIsLoading(false);
    });

    websocketService.on('chat_error', (data) => {
      addMessage({
        content: `Error: ${data.message}`,
        role: 'assistant',
        agent: 'system',
        metadata: { error: true }
      });
      setIsLoading(false);
    });

    return () => {
      websocketService.off('chat_response');
      websocketService.off('chat_error');
    };
  }, [addMessage, selectedAgent]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    // Add user message
    addMessage({
      content: input,
      role: 'user',
      agent: selectedAgent
    });

    // Send to backend
    websocketService.sendChatMessage(input, selectedAgent);
    
    setInput('');
    setIsLoading(true);
  };

  const formatTimestamp = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* Agent Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-foreground mb-2">
          Select AI Agent
        </label>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
        >
          {agents.map((agent) => (
            <option key={agent.id} value={agent.id}>
              {agent.name} - {agent.description}
            </option>
          ))}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4 bg-muted/30 rounded-lg">
        {messages.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">Welcome to PyGent Factory</p>
            <p className="text-sm">Start a conversation with our AI agents</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start space-x-3 ${
                message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                message.role === 'user' 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-secondary text-secondary-foreground'
              }`}>
                {message.role === 'user' ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </div>
              
              <div className={`flex-1 max-w-[80%] ${
                message.role === 'user' ? 'text-right' : ''
              }`}>
                <div className={`inline-block p-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-card border border-border'
                }`}>
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  {message.agent && message.role === 'assistant' && (
                    <p className="text-xs opacity-70 mt-1">
                      via {agents.find(a => a.id === message.agent)?.name || message.agent}
                    </p>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {formatTimestamp(message.timestamp)}
                </p>
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center">
              <Bot className="w-4 h-4" />
            </div>
            <div className="flex-1">
              <div className="inline-block p-3 rounded-lg bg-card border border-border">
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm text-muted-foreground">
                    {agents.find(a => a.id === selectedAgent)?.name} is thinking...
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="mt-4 flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Message ${agents.find(a => a.id === selectedAgent)?.name}...`}
          className="flex-1 p-3 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-4 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>
    </div>
  );
};