# PyGent Factory Status Console

## Overview

The PyGent Factory Status Console is a lightweight headless monitoring tool that provides real-time status information about your PyGent Factory system. It's designed to be run **after** the backend is fully started and operational.

## Features

- **🏢 Backend Status**: Shows if the backend API is responding and healthy
- **🤖 Agent Statistics**: Displays total agents, active/inactive counts, and breakdown by type
- **🔌 MCP Server Status**: Shows total MCP servers, online/offline counts, and individual server status
- **📊 System Information**: Displays uptime, version, and environment information
- **🎨 Colorized Output**: Uses ANSI colors for easy visual status recognition
- **⚡ Real-time Updates**: Automatically refreshes every few seconds

## Usage

### Prerequisites
1. **Backend must be running**: Start your PyGent Factory backend first
2. **Python dependencies**: `aiohttp` is required (automatically installed)

### Running the Status Console

#### Option 1: Quick Start (Windows)
```bash
run_status_console.bat
```

#### Option 2: Manual Command
```bash
python status_console.py [options]
```

#### Available Options
- `--backend-url`: Backend URL (default: http://localhost:8000)
- `--refresh-interval`: Refresh interval in seconds (default: 5)

### Examples

```bash
# Default settings (localhost:8000, 5-second refresh)
python status_console.py

# Custom backend URL
python status_console.py --backend-url http://192.168.1.100:8000

# Faster refresh rate
python status_console.py --refresh-interval 2

# Monitor remote instance
python status_console.py --backend-url https://my-pygent.com --refresh-interval 10
```

## Display Layout

The console displays four main sections:

### 🏢 Backend Status
- Backend API availability
- Database connection status
- Vector Store health
- Memory system status

### 🤖 Agent Statistics
- Total number of agents
- Active vs inactive agents
- Breakdown by agent type (reasoning, search, coding, etc.)

### 🔌 MCP Servers
- Total registered MCP servers
- Online vs offline servers
- List of first 5 servers with their status

### 📊 System Information
- System uptime
- PyGent Factory version
- Environment (development/production)

## Status Indicators

- **🟢 ● ONLINE**: Service is healthy and responding
- **🔴 ● OFFLINE**: Service is unavailable or unhealthy
- **🟡 Numbers**: Warning/informational (e.g., 0 active agents)
- **🟢 Numbers**: Healthy counts (e.g., active agents > 0)

## Workflow

1. **Start Backend**: Ensure PyGent Factory backend is running on port 8000
2. **Run Status Console**: Execute the status console to monitor the system
3. **Monitor**: Watch real-time updates of system health and metrics
4. **Exit**: Press `Ctrl+C` to stop monitoring

## Troubleshooting

### "Failed to fetch status"
- Ensure the backend is running and accessible
- Check the backend URL is correct
- Verify firewall/network connectivity

### No Agent Data
- Agents may not be created yet (normal for fresh installs)
- Check if agents are properly registered in the database

### MCP Servers Offline
- MCP servers may fail to start (common during development)
- Check backend logs for specific MCP server errors

## API Endpoints Used

The status console connects to these backend endpoints:
- `GET /health` - Overall system health
- `GET /api/v1/agents` - Agent information
- `GET /api/v1/mcp/servers` - MCP server status

## Notes

- **Lightweight**: The status console doesn't initialize complex services
- **Read-only**: Only monitors, doesn't modify system state
- **Non-blocking**: Runs independently of the main system
- **Error-resilient**: Continues running even if some endpoints fail

## Integration

The status console is perfect for:
- **Development monitoring**: Keep an eye on system health while coding
- **Production monitoring**: Basic health dashboard for operators
- **Debugging**: Quick overview of system state during troubleshooting
- **Demos**: Show system status during presentations

---

**Remember**: Always start the PyGent Factory backend first, then run the status console to monitor it!
