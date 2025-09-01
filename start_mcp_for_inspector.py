#!/usr/bin/env python3
"""
Startup script for MCP Inspector - properly sets up Python path
"""

import sys
import os
import traceback

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("🚀 Starting YouTube MCP Server...", file=sys.stderr)
    print(f"📁 Python path: {sys.path[0]}", file=sys.stderr)
    
    # Import and run the MCP server
    from youtube_mcp import main
    
    print("✅ MCP server imported successfully", file=sys.stderr)
    print("🔄 Starting MCP server...", file=sys.stderr)
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"❌ Error starting MCP server: {e}", file=sys.stderr)
    print(f"📋 Traceback: {traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)
