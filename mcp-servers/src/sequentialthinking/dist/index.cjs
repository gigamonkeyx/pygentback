#!/usr/bin/env node

/**
 * Simple Sequential Thinking MCP Server
 * Provides tools for structured thinking and reasoning chains
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

// Storage for thinking chains
const thinkingChains = new Map();

class SequentialThinkingServer {
  constructor() {
    this.server = new Server(
      {
        name: 'sequential-thinking-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'start_thinking_chain',
            description: 'Start a new thinking chain for a problem',
            inputSchema: {
              type: 'object',
              properties: {
                problem: {
                  type: 'string',
                  description: 'The problem to think about',
                },
                chain_id: {
                  type: 'string',
                  description: 'Unique identifier for this thinking chain',
                },
              },
              required: ['problem', 'chain_id'],
            },
          },
          {
            name: 'add_thinking_step',
            description: 'Add a step to an existing thinking chain',
            inputSchema: {
              type: 'object',
              properties: {
                chain_id: {
                  type: 'string',
                  description: 'The thinking chain to add to',
                },
                step: {
                  type: 'string',
                  description: 'The thinking step to add',
                },
                step_type: {
                  type: 'string',
                  description: 'Type of thinking step (analysis, hypothesis, conclusion, etc.)',
                  enum: ['analysis', 'hypothesis', 'evidence', 'conclusion', 'question', 'assumption'],
                },
              },
              required: ['chain_id', 'step', 'step_type'],
            },
          },
          {
            name: 'get_thinking_chain',
            description: 'Retrieve a complete thinking chain',
            inputSchema: {
              type: 'object',
              properties: {
                chain_id: {
                  type: 'string',
                  description: 'The thinking chain to retrieve',
                },
              },
              required: ['chain_id'],
            },
          },
          {
            name: 'list_thinking_chains',
            description: 'List all thinking chains',
            inputSchema: {
              type: 'object',
              properties: {},
            },
          },
        ],
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      switch (request.params.name) {
        case 'start_thinking_chain':
          const { problem, chain_id } = request.params.arguments;
          thinkingChains.set(chain_id, {
            problem,
            steps: [],
            created_at: new Date().toISOString(),
          });
          return {
            content: [
              {
                type: 'text',
                text: `Started thinking chain "${chain_id}" for problem: ${problem}`,
              },
            ],
          };

        case 'add_thinking_step':
          const { chain_id: addChainId, step, step_type } = request.params.arguments;
          const chain = thinkingChains.get(addChainId);
          if (!chain) {
            throw new Error(`Thinking chain "${addChainId}" not found`);
          }
          chain.steps.push({
            step,
            step_type,
            timestamp: new Date().toISOString(),
          });
          return {
            content: [
              {
                type: 'text',
                text: `Added ${step_type} step to chain "${addChainId}": ${step}`,
              },
            ],
          };

        case 'get_thinking_chain':
          const { chain_id: getChainId } = request.params.arguments;
          const retrievedChain = thinkingChains.get(getChainId);
          if (!retrievedChain) {
            throw new Error(`Thinking chain "${getChainId}" not found`);
          }
          
          let chainText = `Thinking Chain: ${getChainId}\n`;
          chainText += `Problem: ${retrievedChain.problem}\n`;
          chainText += `Created: ${retrievedChain.created_at}\n\n`;
          chainText += `Steps:\n`;
          
          retrievedChain.steps.forEach((step, index) => {
            chainText += `${index + 1}. [${step.step_type.toUpperCase()}] ${step.step}\n`;
          });
          
          return {
            content: [
              {
                type: 'text',
                text: chainText,
              },
            ],
          };

        case 'list_thinking_chains':
          const chains = Array.from(thinkingChains.entries()).map(([id, chain]) => ({
            id,
            problem: chain.problem,
            steps_count: chain.steps.length,
            created_at: chain.created_at,
          }));
          
          return {
            content: [
              {
                type: 'text',
                text: chains.length > 0 
                  ? JSON.stringify(chains, null, 2)
                  : 'No thinking chains found',
              },
            ],
          };

        default:
          throw new Error(`Unknown tool: ${request.params.name}`);
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Sequential Thinking MCP server running on stdio');
  }
}

const server = new SequentialThinkingServer();
server.run().catch(console.error);
