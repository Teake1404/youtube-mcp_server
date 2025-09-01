# YouTube MCP Server

A powerful MCP (Model Context Protocol) server that provides YouTube API functionality to Claude Desktop and other MCP clients.

## 🚀 Features

- **Video Search**: Search YouTube videos by keywords with filtering by date, views, and popularity
- **Channel Analysis**: Get videos from specific channels with detailed statistics
- **Transcript Extraction**: Download and analyze video transcripts in multiple languages
- **AI Summarization**: Use Claude AI to create intelligent summaries of video content
- **MCP Integration**: Seamlessly connect to Claude Desktop for enhanced AI workflows

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- YouTube Data API v3 key
- Claude Desktop (for MCP integration)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd youtube_mcp
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install google-api-python-client mcp python-dotenv youtube-transcript-api anthropic openai
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ANTHROPIC_API_KEY=your_claude_api_key_here  # Optional, for AI summarization
   ```

4. **Test the installation:**
   ```bash
   python test_mcp_server.py
   ```

## 🔌 MCP Server Usage

### Start the MCP Server

```bash
# Using the startup script
./start_mcp_server.sh

# Or manually
source .venv/bin/activate
python -m youtube_mcp
```

### Available Tools

- `search_youtube_videos` - Search for videos by keywords with filters
- `get_channel_videos` - Get videos from specific channels
- `resolve_channel` - Resolve channel URLs/handles to IDs
- `get_video_transcript` - Extract video transcripts
- `summarize_video_transcript` - AI-powered transcript summarization
- `ping` - Connection test

## 🖥️ Claude Desktop Integration

### Quick Setup

1. **Install Claude Desktop** from [claude.ai/download](https://claude.ai/download)

2. **Configure MCP Server in Claude Desktop:**
   - Open Settings → Model Settings
   - Scroll to "MCP Servers"
   - Click "Add MCP Server"
   - Fill in:
     - **Name**: `youtube-mcp`
     - **Command**: `python`
     - **Arguments**: `-m youtube_mcp`
     - **Working Directory**: `/Users/shuqingke/Documents/youtube_mcp`
     - **Environment Variables**: `PYTHONPATH: .`

3. **Restart Claude Desktop**

### Example Usage in Claude

Once connected, you can ask Claude to:

- "Find the top 5 videos about machine learning from the last 30 days"
- "Get the transcript for video dQw4w9WgXcQ"
- "What's the channel ID for @3Blue1Brown?"
- "Summarize the key points from this video about AI coding"

## 📚 Command Line Usage

The project also includes a comprehensive command-line interface:

```bash
# Search for videos by niche
python test_youtube_api.py niche "ai coding" --max 5 --days 30 --min-views 10000

# Get channel videos
python test_youtube_api.py channel UC_x5XG1OV2P6uZZ5FSM9Ttw --max 10

# Get video transcript
python test_youtube_api.py transcript dQw4w9WgXcQ --summary --max-points 5

# Resolve channel handle
python test_youtube_api.py resolve "@3Blue1Brown"
```

## 🔧 Configuration

### Environment Variables

- `YOUTUBE_API_KEY`: Required. Your YouTube Data API v3 key
- `ANTHROPIC_API_KEY`: Optional. For AI-powered transcript summarization

### API Quotas

- YouTube Data API v3 has daily quotas
- Transcript API has no quotas but may have rate limits
- Monitor usage in Google Cloud Console

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test MCP server
python test_mcp_server.py

# Test YouTube API functionality
python test_youtube_api.py niche "test" --max 1
```

## 📁 Project Structure

```
youtube_mcp/
├── src/youtube_mcp/
│   └── __init__.py          # MCP server implementation
├── test_mcp_server.py       # MCP server tests
├── test_youtube_api.py      # Command-line interface
├── claude_desktop_config.json # Claude Desktop config
├── start_mcp_server.sh      # Startup script
├── CLAUDE_DESKTOP_SETUP.md  # Detailed setup guide
└── pyproject.toml           # Project dependencies
```

## 🚨 Troubleshooting

### Common Issues

1. **"MCP Server not found"**
   - Verify working directory path in Claude Desktop
   - Check that virtual environment is activated
   - Run `python test_mcp_server.py` to verify server works

2. **"YouTube API key not found"**
   - Ensure `.env` file exists with `YOUTUBE_API_KEY`
   - Verify API key has YouTube Data API v3 enabled

3. **Import errors**
   - Activate virtual environment: `source .venv/bin/activate`
   - Reinstall dependencies: `uv pip install -r requirements.txt`

### Debug Steps

1. Test MCP server manually
2. Check Claude Desktop logs
3. Verify file paths and permissions
4. Ensure all dependencies are installed

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to enhance the functionality.

## 📄 License

This project is open source. See LICENSE file for details.

---

**Happy YouTubing with Claude! 🎬✨**

