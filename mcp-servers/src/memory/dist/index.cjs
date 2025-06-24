#!/usr/bin/env node

/**
 * Simple Memory MCP Server
 * Provides basic memory storage and retrieval capabilities
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

// In-memory storage
const memoryStore = new Map();

class MemoryServer {
  constructor() {
    this.server = new Server(
      {
        name: 'memory-server',
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
            name: 'store_memory',
            description: 'Store a key-value pair in memory',
            inputSchema: {
              type: 'object',
              properties: {
                key: {
                  type: 'string',
                  description: 'The key to store the value under',
                },
                value: {
                  type: 'string',
                  description: 'The value to store',
                },
              },
              required: ['key', 'value'],
            },
          },
          {
            name: 'retrieve_memory',
            description: 'Retrieve a value from memory by key',
            inputSchema: {
              type: 'object',
              properties: {
                key: {
                  type: 'string',
                  description: 'The key to retrieve the value for',
                },
              },
              required: ['key'],
            },
          },
          {
            name: 'list_memory_keys',
            description: 'List all keys in memory',
            inputSchema: {
              type: 'object',
              properties: {},
            },
          },
          {
            name: 'clear_memory',
            description: 'Clear all memory',
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
        case 'store_memory':
          const { key, value } = request.params.arguments;
          memoryStore.set(key, value);
          return {
            content: [
              {
                type: 'text',
                text: `Stored value for key "${key}"`,
              },
            ],
          };

        case 'retrieve_memory':
          const retrieveKey = request.params.arguments.key;
          const retrievedValue = memoryStore.get(retrieveKey);
          if (retrievedValue !== undefined) {
            return {
              content: [
                {
                  type: 'text',
                  text: retrievedValue,
                },
              ],
            };
          } else {
            return {
              content: [
                {
                  type: 'text',
                  text: `No value found for key "${retrieveKey}"`,
                },
              ],
            };
          }

        case 'list_memory_keys':
          const keys = Array.from(memoryStore.keys());
          return {
            content: [
              {
                type: 'text',
                text: keys.length > 0 ? keys.join(', ') : 'No keys in memory',
              },
            ],
          };

        case 'clear_memory':
          memoryStore.clear();
          return {
            content: [
              {
                type: 'text',
                text: 'Memory cleared',
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
    console.error('Memory MCP server running on stdio');
  }
}

const server = new MemoryServer();
server.run().catch(console.error);
