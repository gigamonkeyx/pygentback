#!/usr/bin/env python3
"""
MCP Time Server - Time and date operations
Provides time queries, timezone conversion, and scheduling
"""

import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
import pytz


class MCPTimeServer:
    def __init__(self):
        self.supported_timezones = list(pytz.all_timezones)
    
    def get_current_time(self, tz: Optional[str] = None) -> Dict[str, Any]:
        """Get current time in specified timezone"""
        try:
            if tz:
                if tz in self.supported_timezones:
                    timezone_obj = pytz.timezone(tz)
                    dt = datetime.now(timezone_obj)
                else:
                    return {'error': f'Unsupported timezone: {tz}'}
            else:
                dt = datetime.now(timezone.utc)
            
            return {
                'timestamp': dt.isoformat(),
                'unix_timestamp': dt.timestamp(),
                'timezone': str(dt.tzinfo),
                'formatted': dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'minute': dt.minute,
                'second': dt.second
            }
        except Exception as e:
            return {'error': str(e)}
    
    def convert_timezone(self, timestamp: str, from_tz: str, to_tz: str) -> Dict[str, Any]:
        """Convert time between timezones"""
        try:
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            
            converted = dt.astimezone(to_timezone)
            
            return {
                'original': timestamp,
                'converted': converted.isoformat(),
                'from_timezone': from_tz,
                'to_timezone': to_tz,
                'formatted': converted.strftime('%Y-%m-%d %H:%M:%S %Z')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def format_time(self, timestamp: str, format_string: str) -> Dict[str, Any]:
        """Format time with custom format string"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted = dt.strftime(format_string)
            
            return {
                'original': timestamp,
                'formatted': formatted,
                'format': format_string
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        method = request.get('method')
        
        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': 'current_time',
                        'description': 'Get current time in specified timezone',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'timezone': {'type': 'string', 'description': 'Target timezone (optional)'}
                            }
                        }
                    },
                    {
                        'name': 'convert_timezone',
                        'description': 'Convert time between timezones',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'timestamp': {'type': 'string', 'description': 'ISO timestamp'},
                                'from_tz': {'type': 'string', 'description': 'Source timezone'},
                                'to_tz': {'type': 'string', 'description': 'Target timezone'}
                            },
                            'required': ['timestamp', 'from_tz', 'to_tz']
                        }
                    },
                    {
                        'name': 'format_time',
                        'description': 'Format time with custom format string',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'timestamp': {'type': 'string', 'description': 'ISO timestamp'},
                                'format': {'type': 'string', 'description': 'Python strftime format'}
                            },
                            'required': ['timestamp', 'format']
                        }
                    }
                ]
            }
        elif method == 'tools/call':
            tool_name = request.get('params', {}).get('name')
            arguments = request.get('params', {}).get('arguments', {})
            
            if tool_name == 'current_time':
                result = self.get_current_time(arguments.get('timezone'))
            elif tool_name == 'convert_timezone':
                result = self.convert_timezone(
                    arguments.get('timestamp'),
                    arguments.get('from_tz'),
                    arguments.get('to_tz')
                )
            elif tool_name == 'format_time':
                result = self.format_time(
                    arguments.get('timestamp'),
                    arguments.get('format')
                )
            else:
                return {'error': f'Unknown tool: {tool_name}'}
                
            return {'content': [{'type': 'text', 'text': json.dumps(result, indent=2)}]}
        
        return {'error': f'Unknown method: {method}'}


async def main():
    server = MCPTimeServer()
    
    try:
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)


if __name__ == '__main__':
    asyncio.run(main())
