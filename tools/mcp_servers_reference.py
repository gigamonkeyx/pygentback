"""
Official MCP Servers Reference
Source: https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#%EF%B8%8F-official-integrations

This file contains the official list of MCP servers for easy reference and installation.
"""

OFFICIAL_MCP_SERVERS = {
    # Development & Code Tools
    "filesystem": {
        "package": "@modelcontextprotocol/server-filesystem",
        "description": "Secure file system operations with configurable access controls",
        "install_cmd": "npm install -g @modelcontextprotocol/server-filesystem",
        "capabilities": ["file-read", "file-write", "directory-list", "file-search"],
        "usage": "File operations, code editing, project management",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
        }
    },
    
    "github": {
        "package": "@modelcontextprotocol/server-github",
        "description": "GitHub repository integration with search, file operations, and PR management",
        "install_cmd": "npm install -g @modelcontextprotocol/server-github",
        "capabilities": ["repository-search", "file-operations", "issue-management", "pr-management"],
        "usage": "GitHub integration, code search, repository management",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"}
        }
    },
    
    "git": {
        "package": "@modelcontextprotocol/server-git",
        "description": "Git repository operations and version control",
        "install_cmd": "npm install -g @modelcontextprotocol/server-git",
        "capabilities": ["git-operations", "version-control", "branch-management"],
        "usage": "Git operations, version control, commit management",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/path/to/repo"]
        }
    },
    
    # Documentation & Search
    "brave-search": {
        "package": "@modelcontextprotocol/server-brave-search",
        "description": "Web search capabilities using Brave Search API",
        "install_cmd": "npm install -g @modelcontextprotocol/server-brave-search",
        "capabilities": ["web-search", "real-time-info"],
        "usage": "Web search, current information lookup",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "your_api_key_here"}
        }
    },
    
    "puppeteer": {
        "package": "@modelcontextprotocol/server-puppeteer",
        "description": "Web automation and scraping using Puppeteer",
        "install_cmd": "npm install -g @modelcontextprotocol/server-puppeteer",
        "capabilities": ["web-scraping", "automation", "screenshot"],
        "usage": "Web scraping, automated testing, page screenshots",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
        }
    },
    
    # Data & Analytics
    "sqlite": {
        "package": "@modelcontextprotocol/server-sqlite",
        "description": "SQLite database operations and queries",
        "install_cmd": "npm install -g @modelcontextprotocol/server-sqlite",
        "capabilities": ["database-queries", "data-management"],
        "usage": "Database operations, data analysis, SQL queries",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sqlite", "/path/to/database.db"]
        }
    },
    
    "postgres": {
        "package": "@modelcontextprotocol/server-postgres",
        "description": "PostgreSQL database integration",
        "install_cmd": "npm install -g @modelcontextprotocol/server-postgres",
        "capabilities": ["database-queries", "advanced-sql"],
        "usage": "PostgreSQL operations, complex queries, data management",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "env": {"POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost/db"}
        }
    },
    
    # Memory & Context
    "memory": {
        "package": "@modelcontextprotocol/server-memory",
        "description": "Persistent memory and knowledge management",
        "install_cmd": "npm install -g @modelcontextprotocol/server-memory",
        "capabilities": ["persistent-memory", "knowledge-base"],
        "usage": "Long-term memory, knowledge storage, context persistence",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"]
        }
    },
    
    # Time & Scheduling
    "time": {
        "package": "@modelcontextprotocol/server-time",
        "description": "Time and date operations, timezone handling",
        "install_cmd": "npm install -g @modelcontextprotocol/server-time",
        "capabilities": ["time-operations", "timezone-conversion"],
        "usage": "Time calculations, scheduling, timezone conversions",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-time"]
        }
    },
    
    # Cloud & Services
    "aws-kb": {
        "package": "@modelcontextprotocol/server-aws-kb",
        "description": "AWS Knowledge Base integration",
        "install_cmd": "npm install -g @modelcontextprotocol/server-aws-kb",
        "capabilities": ["knowledge-retrieval", "aws-integration"],
        "usage": "AWS documentation, cloud service information",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-aws-kb"],
            "env": {"AWS_ACCESS_KEY_ID": "your_key", "AWS_SECRET_ACCESS_KEY": "your_secret"}
        }
    },
    
    "google-drive": {
        "package": "@modelcontextprotocol/server-gdrive",
        "description": "Google Drive file operations",
        "install_cmd": "npm install -g @modelcontextprotocol/server-gdrive",
        "capabilities": ["file-operations", "cloud-storage"],
        "usage": "Google Drive integration, cloud file management",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-gdrive"]
        }
    },
    
    # Communication
    "gmail": {
        "package": "@modelcontextprotocol/server-gmail",
        "description": "Gmail integration for email operations",
        "install_cmd": "npm install -g @modelcontextprotocol/server-gmail",
        "capabilities": ["email-operations", "gmail-integration"],
        "usage": "Email management, Gmail operations",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-gmail"]
        }
    },
    
    "slack": {
        "package": "@modelcontextprotocol/server-slack",
        "description": "Slack workspace integration",
        "install_cmd": "npm install -g @modelcontextprotocol/server-slack",
        "capabilities": ["messaging", "workspace-integration"],
        "usage": "Slack operations, team communication",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "env": {"SLACK_BOT_TOKEN": "your_bot_token"}
        }
    },
    
    # Development Tools
    "everything": {
        "package": "@modelcontextprotocol/server-everything",
        "description": "Windows Everything search integration",
        "install_cmd": "npm install -g @modelcontextprotocol/server-everything",
        "capabilities": ["file-search", "system-search"],
        "usage": "Fast file search on Windows systems",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything"]
        }
    },
    
    "sequential-thinking": {
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "description": "Sequential reasoning and thinking tools",
        "install_cmd": "npm install -g @modelcontextprotocol/server-sequential-thinking",
        "capabilities": ["reasoning", "step-by-step-thinking"],
        "usage": "Complex problem solving, sequential reasoning",
        "config_example": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
        }
    }
}

# Top 10 MCP servers specifically for coding tasks
TOP_CODING_MCP_SERVERS = [
    "filesystem",      # Essential for file operations
    "github",          # Code repository management
    "git",             # Version control
    "brave-search",    # Documentation and reference lookup
    "puppeteer",       # Web scraping for docs/examples
    "sqlite",          # Data storage and analysis
    "memory",          # Context and knowledge persistence
    "sequential-thinking", # Problem solving
    "everything",      # Fast file search (Windows)
    "time"            # Scheduling and time operations
]

def get_install_commands():
    """Get installation commands for all official MCP servers"""
    commands = []
    for server_id, config in OFFICIAL_MCP_SERVERS.items():
        commands.append(config["install_cmd"])
    return commands

def get_coding_servers_config():
    """Get configuration for the top coding MCP servers"""
    config = {}
    for server_id in TOP_CODING_MCP_SERVERS:
        if server_id in OFFICIAL_MCP_SERVERS:
            config[server_id] = OFFICIAL_MCP_SERVERS[server_id]
    return config

def install_coding_servers():
    """Install all top coding MCP servers"""
    import subprocess
    import logging
    
    logger = logging.getLogger(__name__)
    
    for server_id in TOP_CODING_MCP_SERVERS:
        if server_id in OFFICIAL_MCP_SERVERS:
            cmd = OFFICIAL_MCP_SERVERS[server_id]["install_cmd"]
            logger.info(f"Installing {server_id}: {cmd}")
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ {server_id} installed successfully")
                else:
                    logger.error(f"❌ {server_id} installation failed: {result.stderr}")
            except Exception as e:
                logger.error(f"❌ {server_id} installation error: {e}")

if __name__ == "__main__":
    print("Official MCP Servers Reference")
    print("=" * 50)
    print(f"Total servers available: {len(OFFICIAL_MCP_SERVERS)}")
    print(f"Top coding servers: {len(TOP_CODING_MCP_SERVERS)}")
    print("\nTop Coding Servers:")
    for i, server_id in enumerate(TOP_CODING_MCP_SERVERS, 1):
        server = OFFICIAL_MCP_SERVERS[server_id]
        print(f"{i:2d}. {server_id:15s} - {server['description']}")
    
    print("\nTo install all coding servers, run:")
    print("python -c \"from mcp_servers_reference import install_coding_servers; install_coding_servers()\"")
