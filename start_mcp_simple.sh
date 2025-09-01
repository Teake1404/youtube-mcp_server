#!/bin/bash
# Simple startup script for YouTube MCP Server

cd "/Users/shuqingke/Documents/youtube_mcp/youtube_mcp"
source .venv/bin/activate
export PYTHONPATH="/Users/shuqingke/Documents/youtube_mcp/youtube_mcp/src:/Users/shuqingke/Documents/youtube_mcp/youtube_mcp"
python start_mcp_for_inspector.py
