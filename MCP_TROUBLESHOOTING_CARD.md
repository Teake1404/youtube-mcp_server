# MCP Troubleshooting Quick Reference Card

## 🚨 Common Errors & Quick Fixes

| Error Message | Quick Fix | Full Solution |
|---------------|-----------|---------------|
| `No module named 'mcp'` | `uv pip install mcp` | Install MCP package in virtual environment |
| `'FastMCP' object has no attribute 'run_stdio'` | Change `app.run_stdio()` to `app.run()` | Use correct method name for FastMCP |
| `spawn python ENOENT` | Use full path to Python in `.venv/bin/` | Use absolute path to virtual environment Python |
| `Server transport closed unexpectedly` | Check PYTHONPATH in startup script | Ensure src directory is in Python path |
| `Extension not found in installed extensions` | Use shell script approach | Avoid complex config with args/env, use simple shell script |
| `No module named 'your_package'` | `uv pip install your_package` | Install missing dependencies in virtual environment |
| `Permission denied` | `chmod +x script_name.sh` | Make shell scripts executable |
| `uv: command not found` | `brew install uv` | Install uv package manager |
| `No pyproject.toml found` | Create pyproject.toml or use `uv pip install` | uv needs project file or direct install command |

## 🔧 Quick Config Template

```json
{
  "mcpServers": {
    "YourMCP": {
      "command": "/Users/username/path/to/start_mcp_simple.sh"
    }
  }
}
```

## 📁 Essential File Structure

```
project/
├── .venv/                          # Virtual environment
├── src/
│   └── mcp_name/
│       └── __init__.py            # MCP server code
├── start_mcp_simple.sh             # Startup script
└── start_mcp_for_inspector.py      # Inspector script
```

## 🚀 Startup Script Template

```bash
#!/bin/bash
cd "/Users/username/path/to/project"
source .venv/bin/activate
export PYTHONPATH="/Users/username/path/to/project/src:/Users/username/path/to/project"
python start_mcp_for_inspector.py
```

## 🧪 Testing Commands

```bash
# Test locally
python start_mcp_for_inspector.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector stdio --command "/path/to/start_mcp_simple.sh"

# Check if running
ps aux | grep python | grep your_script
```

## 🚀 uv Commands Reference

```bash
# Create virtual environment
uv venv --python 3.11

# Install packages
uv pip install package_name

# Sync from pyproject.toml
uv sync

# Check uv version
uv --version

# List installed packages
uv pip list
```

## ⚡ Quick Fix Checklist

- [ ] Virtual environment activated?
- [ ] MCP package installed? (`uv pip install mcp`)
- [ ] Scripts executable? (`chmod +x *.sh`)
- [ ] PYTHONPATH set correctly?
- [ ] Using `app.run()` not `app.run_stdio()`?
- [ ] Full absolute paths everywhere?
- [ ] Claude Desktop restarted after config change?

## 🎯 Golden Rules

1. **Always use shell scripts** for startup
2. **Set PYTHONPATH in startup script**
3. **Use absolute paths** everywhere
4. **Test locally first**
5. **Restart Claude Desktop** after config changes

---

**Save this card for quick reference! 📋✨**
