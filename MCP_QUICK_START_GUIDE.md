# MCP Quick Start Guide - Build MCPs in Minutes!

This guide captures all the lessons learned from setting up the YouTube MCP server, including troubleshooting steps, configuration patterns, and best practices.

## 🚀 Quick Setup Template

### 1. Project Structure
```
your_mcp_project/
├── .venv/                          # Virtual environment
├── src/
│   └── your_mcp_name/
│       └── __init__.py            # Main MCP server file
├── start_mcp_simple.sh             # Startup script
├── pyproject.toml                  # Dependencies
└── .env                           # Environment variables
```

### 2. Python Dependencies (pyproject.toml)
```toml
[project]
name = "your-mcp-name"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.13.1",                 # Required for MCP server
    "python-dotenv>=1.1.1",        # For environment variables
    # Add your specific dependencies here
]

[build-system]
requires = ["uv_build>=0.8.3,<0.9.0"]
build-backend = "uv_build"
```

### 3. MCP Server Template (src/your_mcp_name/__init__.py)
```python
#!/usr/bin/env python3
"""
Your MCP Server - Provides functionality to Claude Desktop
"""

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
app = FastMCP("your-mcp-name")

@app.tool()
def your_tool_function(param1: str, param2: int = 10) -> dict:
    """
    Description of what your tool does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
    
    Returns:
        Dict containing results
    """
    try:
        # Your tool logic here
        result = f"Processed {param1} with {param2}"
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.tool()
def ping() -> str:
    """Simple ping to test if the MCP server is working"""
    return "pong from Your MCP Server!"

def main() -> None:
    """Run the MCP server"""
    app.run()  # Note: NOT app.run_stdio()

if __name__ == "__main__":
    main()
```

### 4. Startup Script (start_mcp_simple.sh)
```bash
#!/bin/bash
# Startup script for Your MCP Server

cd "/Users/your_username/Documents/your_mcp_project"
source .venv/bin/activate
export PYTHONPATH="/Users/your_username/Documents/your_mcp_project/src:/Users/your_username/Documents/your_mcp_project"
python start_mcp_for_inspector.py
```

### 5. Inspector Script (start_mcp_for_inspector.py)
```python
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
    print("🚀 Starting Your MCP Server...", file=sys.stderr)
    print(f"📁 Python path: {sys.path[0]}", file=sys.stderr)
    
    # Import and run the MCP server
    from your_mcp_name import main
    
    print("✅ MCP server imported successfully", file=sys.stderr)
    print("🔄 Starting MCP server...", file=sys.stderr)
    
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"❌ Error starting MCP server: {e}", file=sys.stderr)
    print(f"📋 Traceback: {traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)
```

## 🔧 Installation Commands

### 1. Install uv (if not already installed)
```bash
# macOS (using Homebrew)
brew install uv

# Or download from https://github.com/astral-sh/uv
```

**Why uv?** It's significantly faster than pip, handles virtual environments better, and provides more reliable dependency resolution for MCP projects.

### 2. Create Virtual Environment
```bash
cd your_mcp_project
uv venv --python 3.11
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core MCP dependencies
uv pip install mcp python-dotenv

# Install your specific packages
uv pip install your-specific-package

# Or use uv sync if you have a pyproject.toml
uv sync
```

### 3. Make Scripts Executable
```bash
chmod +x start_mcp_simple.sh
chmod +x start_mcp_for_inspector.py
```

## 📱 Claude Desktop Configuration

### 1. Config File Location
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### 2. Config Template
```json
{
  "mcpServers": {
    "YourExistingMCP": {
      "command": "/path/to/existing/mcp/start_mcp.sh"
    },
    "YourNewMCP": {
      "command": "/Users/your_username/Documents/your_mcp_project/start_mcp_simple.sh"
    }
  }
}
```

### 3. Important Notes
- **Use shell scripts** instead of direct Python commands (more reliable)
- **Full absolute paths** for all commands
- **No args, cwd, or env** in config - handle everything in the shell script
- **Unique names** for each MCP server

## 🧪 Testing & Debugging

### 1. Test MCP Server Locally
```bash
cd your_mcp_project
source .venv/bin/activate
python start_mcp_for_inspector.py
# Should start without errors
```

### 2. Test with MCP Inspector
```bash
npx @modelcontextprotocol/inspector stdio --command "/Users/your_username/Documents/your_mcp_project/start_mcp_simple.sh"
```

### 3. Test in Claude Desktop
1. Restart Claude Desktop after config changes
2. Check logs for connection status
3. Try using a tool from your MCP

## 🚨 Common Issues & Solutions

### 1. "No module named 'mcp'"
**Solution**: Install MCP package
```bash
uv pip install mcp
```

### 2. "AttributeError: 'FastMCP' object has no attribute 'run_stdio'"
**Solution**: Use `app.run()` not `app.run_stdio()`

### 3. "spawn python ENOENT"
**Solution**: Use full path to Python in virtual environment
```bash
/Users/username/project/.venv/bin/python
```

### 4. "Server transport closed unexpectedly"
**Solution**: Check Python path in startup script and ensure all imports work

### 5. "Extension not found in installed extensions"
**Solution**: Use shell script approach instead of complex config with args/env

## 📋 Checklist for New MCP

- [ ] Create project directory with `src/` structure
- [ ] Set up Python 3.11+ virtual environment
- [ ] Install `mcp` and other dependencies
- [ ] Create MCP server with `FastMCP` and `@app.tool()` decorators
- [ ] Use `app.run()` in main function
- [ ] Create startup script that sets PYTHONPATH
- [ ] Make scripts executable with `chmod +x`
- [ ] Test locally with `python start_mcp_for_inspector.py`
- [ ] Test with MCP Inspector
- [ ] Add to Claude Desktop config using shell script approach
- [ ] Restart Claude Desktop and test tools

## 🎯 Best Practices

1. **Always use shell scripts** for startup (more reliable than complex configs)
2. **Set PYTHONPATH in startup script** (don't rely on config env vars)
3. **Use absolute paths** everywhere (relative paths cause issues)
4. **Test locally first** before adding to Claude Desktop
5. **Keep MCP server names unique** across all your projects
6. **Use Python 3.11+** for compatibility with latest MCP packages
7. **Handle errors gracefully** in your tools with try/catch blocks
8. **Use uv for dependency management** (faster and more reliable than pip)
9. **Create virtual environments with uv** (ensures proper Python version isolation)

## 🔗 Useful Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [FastMCP Examples](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/model-context-protocol-mcp)

---

**Happy MCP Building! 🚀✨**
