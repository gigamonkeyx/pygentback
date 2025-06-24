import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Brain, Dna, Search, Settings, Code, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useChat, useUI } from '@/stores/appStore';
import { websocketService } from '@/services/websocket';
import { ChatMessage, ReasoningMode } from '@/types';
import { MessageList } from './MessageList';
import { AgentSelector } from './AgentSelector';
import { ReasoningPanel } from './ReasoningPanel';
import { cn } from '@/utils/cn';

export type AgentType = 'reasoning' | 'evolution' | 'search' | 'general' | 'coding' | 'research';

interface ChatInterfaceProps {
  className?: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ className }) => {
  const [message, setMessage] = useState('');
  const [activeAgent, setActiveAgent] = useState<AgentType>('reasoning');
  const [isConnected, setIsConnected] = useState(false);
  const [showReasoningPanel, setShowReasoningPanel] = useState(false);
  const [lastSentQuery, setLastSentQuery] = useState<string>(''); // Track last sent message
  const [lastAgentResponse, setLastAgentResponse] = useState<string>(''); // Track last response
  const [pipelineLogs, setPipelineLogs] = useState<string[]>([]); // Track pipeline logs
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Helper function to add pipeline logs
  const addPipelineLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    setPipelineLogs(prev => [...prev.slice(-10), logEntry]); // Keep last 10 logs
    console.log(`🔍 PIPELINE: ${logEntry}`);
  };

  // Add a test log on component mount
  useEffect(() => {
    addPipelineLog('🟢 Pipeline logger initialized');
    addPipelineLog('🔧 Testing log display functionality');
  }, []);

  const {
    conversations,
    activeConversation,
    typingUsers,
    addMessage,
    setTypingUser
  } = useChat();

  const { setLoading } = useUI();

  const currentMessages = conversations.get(activeConversation) || [];

  useEffect(() => {
    // Check WebSocket connection status (connection is handled in App.tsx)
    const checkConnection = () => {
      setIsConnected(websocketService.isConnected());
    };

    // Check if already connected (connection is handled in App.tsx)
    checkConnection();
    setLoading('chat', false);

    // Set up event listeners
    const unsubscribeResponse = websocketService.on('chat_response', (data: any) => {
      if (data.message) {
        addMessage(activeConversation, data.message);
      }
    });

    const unsubscribeTyping = websocketService.on('typing_indicator', (data: any) => {
      setTypingUser(data.user_id, data.typing);
    });

    const unsubscribeConnection = websocketService.on('connection_status', (data: any) => {
      setIsConnected(data.connected);
    });

    return () => {
      unsubscribeResponse();
      unsubscribeTyping();
      unsubscribeConnection();
    };
  }, [activeConversation, addMessage, setTypingUser, setLoading]);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentMessages]);

  const sendMessage = async () => {
    const userQuery = message.trim();
    if (!userQuery) return;
    
    // Track the query being sent (do this FIRST, even if not connected)
    setLastSentQuery(userQuery);
    addPipelineLog(`📤 Frontend: User query captured: "${userQuery}"`);

    // DEBUG: Log original user query
    console.log('🟢 ORIGINAL USER QUERY:', userQuery);
    console.log('🔗 CONNECTION STATUS:', isConnected);

    // If not connected, still show what we tried to send
    if (!isConnected) {
      addPipelineLog('❌ WebSocket disconnected - message not sent');
      console.warn('⚠️ NOT CONNECTED - Message not sent but tracked for debug');
      return;
    }

    addPipelineLog('🔗 WebSocket connected - preparing message');

    const chatMessage: ChatMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'user',
      content: userQuery,
      timestamp: new Date(),
      metadata: {
        reasoning_mode: getReasoningModeForAgent(activeAgent)
      }
    };

    // DEBUG: Log what we're adding to UI
    console.log('🟡 CHAT MESSAGE FOR UI:', chatMessage);

    // Add user message immediately
    addMessage(activeConversation, chatMessage);
    addPipelineLog('💬 Message added to UI chat');

    try {
      addPipelineLog(`🚀 Sending to backend: agent=${activeAgent}`);

      // DEBUG: Log what we're sending to backend
      console.log('🔵 MESSAGE SENT TO BACKEND:', chatMessage);
      console.log('🔵 ACTIVE AGENT:', activeAgent);

      // Send message via WebSocket with correct format
      const messageToSend = {
        ...chatMessage,
        agentId: activeAgent
      };
      
      // Log what we're actually sending
      console.log('🔵 EXACT MESSAGE TO BACKEND:', messageToSend);
      
      websocketService.sendMessage('chat_message', {
        message: messageToSend,
        timestamp: new Date().toISOString()
      });
      addPipelineLog('📡 WebSocket message sent to backend');

      // Clear input
      setMessage('');

      // Show reasoning panel for reasoning agent
      if (activeAgent === 'reasoning') {
        setShowReasoningPanel(true);
      }

    } catch (error) {
      addPipelineLog(`❌ Error: ${error}`);
      console.error('❌ FAILED TO SEND MESSAGE:', error);
      // TODO: Add error notification
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getReasoningModeForAgent = (agent: AgentType): ReasoningMode => {
    switch (agent) {
      case 'reasoning':
        return ReasoningMode.TOT_ENHANCED_RAG;
      case 'evolution':
        return ReasoningMode.ADAPTIVE;
      case 'search':
        return ReasoningMode.RAG_ONLY;
      case 'coding':
        return ReasoningMode.ADAPTIVE;
      case 'research':
        return ReasoningMode.RAG_ONLY;
      default:
        return ReasoningMode.ADAPTIVE;
    }
  };

  const getAgentIcon = (agent: AgentType) => {
    switch (agent) {
      case 'reasoning':
        return <Brain className="h-4 w-4" />;
      case 'evolution':
        return <Dna className="h-4 w-4" />;
      case 'search':
        return <Search className="h-4 w-4" />;
      case 'coding':
        return <Code className="h-4 w-4" />;
      case 'research':
        return <BookOpen className="h-4 w-4" />;
      default:
        return <Bot className="h-4 w-4" />;
    }
  };

  const getAgentDescription = (agent: AgentType) => {
    switch (agent) {
      case 'reasoning':
        return 'Advanced Tree of Thought reasoning with multi-path exploration';
      case 'evolution':
        return 'AI-guided recipe evolution and optimization';
      case 'search':
        return 'GPU-accelerated vector search and document retrieval';
      case 'coding':
        return 'Expert code generation, analysis, debugging, and optimization';
      case 'research':
        return 'Academic and historical research with literature review capabilities';
      default:
        return 'General AI assistant for various tasks';
    }
  };

  return (
    <div className={cn('flex h-full', className)}>
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        {/* Chat Header */}
        <Card className="rounded-none border-x-0 border-t-0">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  {getAgentIcon(activeAgent)}
                  <CardTitle className="text-lg">
                    {activeAgent.charAt(0).toUpperCase() + activeAgent.slice(1)} Agent
                  </CardTitle>
                </div>
                <div className={cn(
                  'h-2 w-2 rounded-full',
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                )} />
              </div>
              <div className="flex items-center space-x-2">
                <AgentSelector
                  value={activeAgent}
                  onChange={setActiveAgent}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowReasoningPanel(!showReasoningPanel)}
                  className={cn(
                    activeAgent === 'reasoning' ? 'visible' : 'invisible'
                  )}
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              {getAgentDescription(activeAgent)}
            </p>
            
            {/* PROMINENT Debug Information Panel */}
            <div className="bg-red-100 dark:bg-red-900 border-2 border-red-500 p-4 rounded-lg mt-3 text-sm">
              <div className="font-bold mb-3 text-red-800 dark:text-red-200">🚨 DEBUG MODE ACTIVE - QUERY PIPELINE MONITORING</div>
              <div className="space-y-2">
                <div>Active Agent: <span className="font-mono bg-blue-200 dark:bg-blue-800 px-2 py-1 rounded text-black dark:text-white">{activeAgent}</span></div>
                <div>WebSocket Status: <span className={`font-mono px-2 py-1 rounded text-black dark:text-white ${isConnected ? 'bg-green-200 dark:bg-green-800' : 'bg-red-200 dark:bg-red-800'}`}>
                  {isConnected ? '✅ CONNECTED' : '❌ DISCONNECTED (but messages may still work)'}
                </span></div>
                <div>Active Conversation: <span className="font-mono bg-purple-200 dark:bg-purple-800 px-2 py-1 rounded text-black dark:text-white">{activeConversation}</span></div>
                
                <div className="bg-orange-100 dark:bg-orange-900 p-2 rounded border-2 border-orange-500">
                  <div className="font-bold text-orange-800 dark:text-orange-200">📤 Last Query Sent:</div>
                  <div className="font-mono text-xs bg-white dark:bg-gray-800 p-1 rounded mt-1 text-black dark:text-white">
                    {lastSentQuery || 'No query sent yet - type and send a message above'}
                  </div>
                </div>
                
                <div className="bg-yellow-200 dark:bg-yellow-800 p-2 rounded mt-2">
                  <div className="font-bold text-yellow-800 dark:text-yellow-200">⚠️ AGENT GETTING MOCK RESPONSES</div>
                  <div className="text-yellow-700 dark:text-yellow-300 text-xs mt-1">Send a message above to see the query that gets sent to the agent</div>
                </div>
              </div>
            </div>
          </CardHeader>
        </Card>

        {/* Query Pipeline Log Window */}
        <Card className="rounded-none border-x-0">
          <CardHeader className="pb-2">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm text-red-600 dark:text-red-400">🔍 Live Query Pipeline Log</CardTitle>
              <button 
                onClick={() => addPipelineLog('🧪 Manual test log entry')}
                className="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600"
              >
                Test Log
              </button>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-32 bg-black text-green-400 font-mono text-xs p-3 rounded border overflow-y-auto">
              <div className="space-y-1">
                <div className="text-yellow-400">[PIPELINE MONITOR ACTIVE - Count: {pipelineLogs.length}]</div>
                {pipelineLogs.length === 0 ? (
                  <>
                    <div>📤 Frontend: Ready to capture queries</div>
                    <div>🔗 WebSocket: {isConnected ? '✅ Connected' : '❌ Disconnected'}</div>
                    <div>🤖 Backend: Ready to process messages</div>
                    <div className="text-blue-400">Send a message above to see real-time pipeline trace</div>
                  </>
                ) : (
                  pipelineLogs.map((log, index) => (
                    <div key={index} className="text-green-300">{log}</div>
                  ))
                )}
                <div className="text-gray-500">Check browser console (F12) and backend terminal for full logs</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          <MessageList
            messages={currentMessages}
            typingUsers={typingUsers}
            className="h-full"
          />
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input */}
        <Card className="rounded-none border-x-0 border-b-0">
          <CardContent className="p-4">
            <div className="flex space-x-2">
              <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={`Ask the ${activeAgent} agent...`}
                disabled={false}
                className="flex-1"
                aria-label="Chat message input"
                data-testid="chat-input"
                id="chat-input"
              />
              <Button
                onClick={sendMessage}
                disabled={!message.trim() || !isConnected}
                size="icon"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            {!isConnected && (
              <p className="text-sm text-yellow-600 mt-2">
                ⚠️ Running in offline mode - real-time features unavailable
              </p>
            )}
            {typingUsers.size > 0 && (
              <p className="text-sm text-muted-foreground mt-2">
                AI agent is thinking...
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Reasoning Panel */}
      {showReasoningPanel && activeAgent === 'reasoning' && (
        <div className="w-96 border-l">
          <ReasoningPanel
            onClose={() => setShowReasoningPanel(false)}
          />
        </div>
      )}
    </div>
  );
};
