# YouTube MCP Server - Claude Desktop Setup Guide

This guide will help you connect your YouTube MCP server to Claude Desktop so you can use YouTube tools directly within Claude.

## 🚀 Quick Start

1. **Test your MCP server** (make sure it works):
   ```bash
   cd youtube_mcp
   python test_mcp_server.py
   ```

2. **Install Claude Desktop** (if not already installed):
   - Download from: https://claude.ai/download
   - Install and launch Claude Desktop

3. **Configure Claude Desktop** to connect to your MCP server

## 📋 Prerequisites

- ✅ YouTube API key in `.env` file
- ✅ Python 3.11+ with all dependencies installed
- ✅ Claude Desktop installed
- ✅ MCP server tested and working

## 🔧 Claude Desktop Configuration

### Option 1: Direct Configuration (Recommended)

1. **Open Claude Desktop**
2. **Go to Settings** (gear icon in bottom left)
3. **Click on "Model Settings"**
4. **Scroll down to "MCP Servers"**
5. **Click "Add MCP Server"**
6. **Fill in the details:**
   - **Name**: `youtube-mcp`
   - **Command**: `python`
   - **Arguments**: `-m youtube_mcp`
   - **Working Directory**: `/Users/shuqingke/Documents/youtube_mcp` (your project path)
   - **Environment Variables**: 
     - `PYTHONPATH`: `.`

### Option 2: Configuration File

1. **Copy the configuration file** to Claude Desktop's config directory:
   ```bash
   # On macOS, the config directory is typically:
   cp claude_desktop_config.json ~/Library/Application\ Support/Claude\ Desktop/
   ```

2. **Restart Claude Desktop**

## 🧪 Testing the Connection

1. **Restart Claude Desktop** after configuration
2. **Start a new conversation**
3. **Ask Claude to use a YouTube tool**, for example:
   ```
   Can you search for videos about "AI coding" using the YouTube tools?
   ```

4. **Claude should now have access to these YouTube tools:**
   - `search_youtube_videos` - Search for videos by keywords
   - `get_channel_videos` - Get videos from a specific channel
   - `resolve_channel` - Resolve channel URLs/handles to IDs
   - `get_video_transcript` - Get video transcripts
   - `summarize_video_transcript` - Get transcripts and AI summaries

## 🔍 Troubleshooting

### Common Issues

1. **"MCP Server not found"**
   - Check that the working directory path is correct
   - Make sure you're running from the project root
   - Verify the MCP server can start manually

2. **"Permission denied"**
   - Make sure the Python path is correct
   - Check that all dependencies are installed
   - Try running `python test_mcp_server.py` first

3. **"YouTube API key not found"**
   - Verify your `.env` file exists and contains `YOUTUBE_API_KEY`
   - Make sure the API key is valid and has YouTube Data API v3 enabled

### Debug Steps

1. **Test MCP server manually:**
   ```bash
   cd youtube_mcp
   python test_mcp_server.py
   ```

2. **Check Claude Desktop logs:**
   - Look for MCP-related errors in the console
   - Check the Claude Desktop application logs

3. **Verify file paths:**
   - Make sure all paths in the configuration are absolute
   - Check that the working directory exists and contains your code

## 📚 Example Usage

Once connected, you can ask Claude to:

- **Search for videos**: "Find the top 5 videos about machine learning from the last 30 days"
- **Get channel info**: "What's the channel ID for @3Blue1Brown?"
- **Get transcripts**: "Get the transcript for video dQw4w9WgXcQ"
- **Summarize content**: "Summarize the key points from this video transcript"

## 🎯 Next Steps

1. **Test basic functionality** with simple searches
2. **Try transcript features** with a video you know
3. **Experiment with different parameters** (date ranges, view counts, etc.)
4. **Customize the tools** if you need additional functionality

## 📞 Getting Help

If you encounter issues:

1. **Check the test output** from `test_mcp_server.py`
2. **Verify your configuration** matches the examples above
3. **Check Claude Desktop logs** for error messages
4. **Ensure all dependencies** are properly installed

---

**Happy YouTubing with Claude! 🎬✨**

