#!/usr/bin/env python3
"""
YouTube MCP Server - Provides YouTube API functionality to Claude Desktop
"""

import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize FastMCP
app = FastMCP("youtube-mcp")

# Global YouTube service
_youtube_service = None

def get_youtube_service():
    """Get or create YouTube service instance"""
    global _youtube_service
    if _youtube_service is None:
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment variables")
        _youtube_service = build('youtube', 'v3', developerKey=api_key)
    return _youtube_service

def resolve_channel_id(url_or_handle: str) -> str:
    """Resolve a channel URL or @handle to a UC... channel ID."""
    youtube = get_youtube_service()
    text = url_or_handle.strip()
    
    # Extract handle from common URL formats
    if 'youtube.com' in text and '/@' in text:
        text = text.split('/@', 1)[1]
    if text.startswith('@'):
        text = text[1:]

    # Search for the channel by handle/name
    resp = youtube.search().list(
        part='snippet',
        q='@' + text,
        type='channel',
        maxResults=1,
    ).execute()
    items = resp.get('items', [])
    if not items:
        # Fallback: try without @
        resp = youtube.search().list(
            part='snippet', q=text, type='channel', maxResults=1
        ).execute()
        items = resp.get('items', [])
    if not items:
        raise ValueError(f"Could not resolve channel from: {url_or_handle}")
    return items[0]['snippet']['channelId']

def _get_video_stats(video_ids: List[str]) -> Dict[str, Dict]:
    """Get statistics for multiple videos"""
    youtube = get_youtube_service()
    stats: Dict[str, Dict] = {}
    
    # videos.list supports up to 50 IDs at a time
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        vresp = youtube.videos().list(
            part='statistics,snippet', id=','.join(batch)
        ).execute()
        for v in vresp.get('items', []):
            stats[v['id']] = {
                'view_count': int(v['statistics'].get('viewCount', 0)),
                'like_count': int(v['statistics'].get('likeCount', 0)) if 'likeCount' in v['statistics'] else None,
                'published_at': v['snippet'].get('publishedAt'),
                'title': v['snippet'].get('title'),
                'channel_title': v['snippet'].get('channelTitle'),
            }
    return stats

@app.tool()
def search_youtube_videos(
    query: str,
    max_results: int = 10,
    days: Optional[int] = 90,
    min_views: Optional[int] = 20000
) -> List[Dict[str, Any]]:
    """
    Search for YouTube videos by niche keyword(s) with popularity/recency filters.
    
    Args:
        query: Keyword(s) to search for, e.g. "ai coding"
        max_results: Maximum number of results to return (default: 10)
        days: Only include videos published in the last N days (default: 90)
        min_views: Only include videos with at least this many views (default: 20000)
    
    Returns:
        List of video information including title, channel, views, and URL
    """
    try:
        youtube = get_youtube_service()
        
        published_after = None
        if days and days > 0:
            dt = datetime.now(timezone.utc) - timedelta(days=days)
            published_after = dt.isoformat()

        # Pull more than needed to allow filtering by views
        resp = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            order='viewCount',
            publishedAfter=published_after,
            maxResults=min(50, max(10, max_results * 2)),
        ).execute()
        
        ids = [it['id']['videoId'] for it in resp.get('items', [])]
        stats = _get_video_stats(ids)

        candidates: List[tuple[int, Dict]] = []
        for vid, meta in stats.items():
            views = meta['view_count'] or 0
            if min_views is not None and views < min_views:
                continue
            candidates.append((views, {
                'video_id': vid,
                'title': meta['title'],
                'channel_title': meta['channel_title'],
                'published_at': meta['published_at'],
                'view_count': views,
                'url': f"https://youtu.be/{vid}"
            }))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in candidates[:max_results]]
        
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

@app.tool()
def get_channel_videos(
    channel_id: str,
    max_results: int = 10,
    title: Optional[str] = None,
    days: Optional[int] = 90,
    min_views: Optional[int] = 20000
) -> List[Dict[str, Any]]:
    """
    Get videos from a specific YouTube channel with optional filtering.
    
    Args:
        channel_id: YouTube channel ID (starts with UC...)
        max_results: Maximum number of results to return (default: 10)
        title: Specific video title to find within the channel (optional)
        days: Only include videos published in the last N days (default: 90)
        min_views: Only include videos with at least this many views (default: 20000)
    
    Returns:
        List of video information from the channel
    """
    try:
        youtube = get_youtube_service()
        
        # If a specific title is requested, search within the channel for that title
        if title:
            resp = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                q=title,
                type='video',
                order='relevance',
                maxResults=5,
            ).execute()
            # Prefer exact (case-insensitive) title match if available
            items_raw = resp.get('items', [])
            exact_matches = [i for i in items_raw if i['snippet'].get('title', '').strip().lower() == title.strip().lower()]
            selected = exact_matches[:1] if exact_matches else items_raw[:1]
        else:
            # Otherwise return recent uploads from the channel
            published_after = None
            if days and days > 0:
                dt = datetime.now(timezone.utc) - timedelta(days=days)
                published_after = dt.isoformat()
            resp = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                publishedAfter=published_after,
                maxResults=min(50, max(10, max_results * 2)),
            ).execute()
            selected = resp.get('items', [])

        ids = [it['id']['videoId'] for it in selected]
        stats = _get_video_stats(ids)

        candidates: List[tuple[int, Dict]] = []
        for vid, meta in stats.items():
            views = meta['view_count'] or 0
            if min_views is not None and views < min_views:
                continue
            candidates.append((views, {
                'video_id': vid,
                'title': meta['title'],
                'channel_title': meta['channel_title'],
                'published_at': meta['published_at'],
                'view_count': views,
                'url': f"https://youtu.be/{vid}"
            }))

        # If title search was used, we'll already have 1 best match; otherwise, sort by views desc
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in candidates[:max_results]]
        
    except Exception as e:
        return [{"error": f"Channel search failed: {str(e)}"}]

@app.tool()
def resolve_channel(
    url_or_handle: str
) -> Dict[str, Any]:
    """
    Resolve a channel URL or @handle to a channel ID and basic information.
    
    Args:
        url_or_handle: Channel URL or @handle (e.g., "https://youtube.com/@Handle" or "@Handle")
    
    Returns:
        Channel information including ID, title, and description
    """
    try:
        youtube = get_youtube_service()
        channel_id = resolve_channel_id(url_or_handle)
        
        # Get channel details
        resp = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        ).execute()
        
        if resp.get('items'):
            channel = resp['items'][0]
            snippet = channel['snippet']
            stats = channel.get('statistics', {})
            
            return {
                'channel_id': channel_id,
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'video_count': int(stats.get('videoCount', 0)),
                'view_count': int(stats.get('viewCount', 0)),
                'custom_url': snippet.get('customUrl'),
                'published_at': snippet.get('publishedAt')
            }
        else:
            return {"error": "Channel not found"}
            
    except Exception as e:
        return {"error": f"Failed to resolve channel: {str(e)}"}

@app.tool()
def get_video_transcript(
    video_id: str,
    languages: Optional[List[str]] = None,
    preserve_formatting: bool = False
) -> Dict[str, Any]:
    """
    Get transcript for a YouTube video.
    
    Args:
        video_id: YouTube video ID
        languages: List of language codes to try (default: ['en'])
        preserve_formatting: Whether to preserve HTML formatting
    
    Returns:
        Dict containing transcript data and metadata
    """
    if languages is None:
        languages = ['en']
    
    try:
        ytt_api = YouTubeTranscriptApi()
        
        # First, list available transcripts to see what's available
        transcript_list = ytt_api.list(video_id)
        
        # Try to find a transcript in the requested languages
        transcript = transcript_list.find_transcript(languages)
        
        # Fetch the transcript
        fetched_transcript = transcript.fetch()
        
        # Get raw data
        raw_data = fetched_transcript.to_raw_data()
        
        # Combine all text snippets
        full_text = ' '.join([snippet['text'] for snippet in raw_data])
        
        return {
            'success': True,
            'video_id': video_id,
            'language': transcript.language,
            'language_code': transcript.language_code,
            'is_generated': transcript.is_generated,
            'snippet_count': len(raw_data),
            'full_transcript': full_text,
            'raw_snippets': raw_data,
            'metadata': {
                'total_duration': sum(snippet.get('duration', 0) for snippet in raw_data),
                'avg_snippet_length': len(full_text) / len(raw_data) if raw_data else 0
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'video_id': video_id,
            'error': str(e),
            'error_type': type(e).__name__
        }

@app.tool()
def summarize_video_transcript(
    video_id: str,
    max_points: int = 5,
    use_llm: bool = True,
    llm_provider: str = "claude"
) -> Dict[str, Any]:
    """
    Get transcript and create a summary for a YouTube video.
    
    Args:
        video_id: YouTube video ID
        max_points: Maximum number of summary points to generate
        use_llm: Whether to use LLM for summarization (default: True)
        llm_provider: LLM provider to use ('claude', 'openai', or 'simple')
    
    Returns:
        Dict containing transcript and summary information
    """
    try:
        # First get the transcript
        transcript_data = get_video_transcript(video_id)
        
        if not transcript_data.get('success'):
            return {
                'success': False,
                'error': 'No transcript data available for summarization'
            }
        
        # If LLM is disabled, fall back to simple summarization
        if not use_llm or llm_provider == "simple":
            return _simple_summarization(transcript_data, max_points)
        
        try:
            if llm_provider == "claude":
                return _claude_summarization(transcript_data, max_points)
            else:
                return _simple_summarization(transcript_data, max_points)
                
        except Exception as e:
            return _simple_summarization(transcript_data, max_points)
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Summarization failed: {str(e)}'
        }

def _claude_summarization(transcript_data: Dict, max_points: int) -> Dict:
    """Use Claude API for intelligent summarization"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare the prompt for Claude
        prompt = f"""You are an expert content analyst. Please analyze this YouTube video transcript and provide a comprehensive summary.

Video ID: {transcript_data['video_id']}
Language: {transcript_data['language']}
Duration: {transcript_data['metadata']['total_duration']:.1f} seconds

TRANSCRIPT:
{transcript_data['full_transcript']}

Please provide:
1. A concise executive summary (2-3 sentences)
2. {max_points} key points that capture the main ideas, concepts, or insights
3. The overall tone and style of the content
4. Any technical terms or concepts mentioned
5. Target audience for this content

Format your response as JSON with these keys:
- executive_summary
- key_points (array of strings)
- tone_style
- technical_concepts (array of strings)
- target_audience

Keep each key point concise but informative (max 100 words each)."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Try to parse JSON response
        try:
            import json
            content = response.content[0].text
            # Extract JSON from the response (Claude might wrap it in markdown)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content
            
            summary_data = json.loads(json_str)
            
            return {
                'success': True,
                'video_id': transcript_data['video_id'],
                'summary_method': 'claude_llm',
                'executive_summary': summary_data.get('executive_summary', ''),
                'key_points': summary_data.get('key_points', []),
                'tone_style': summary_data.get('tone_style', ''),
                'technical_concepts': summary_data.get('technical_concepts', []),
                'target_audience': summary_data.get('target_audience', ''),
                'statistics': _calculate_statistics(transcript_data),
                'raw_llm_response': content
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: extract key points from text response
            content = response.content[0].text
            key_points = _extract_key_points_from_text(content, max_points)
            
            return {
                'success': True,
                'video_id': transcript_data['video_id'],
                'summary_method': 'claude_llm_fallback',
                'key_points': key_points,
                'statistics': _calculate_statistics(transcript_data),
                'raw_llm_response': content
            }
            
    except Exception as e:
        raise Exception(f"Claude API error: {str(e)}")

def _simple_summarization(transcript_data: Dict, max_points: int) -> Dict:
    """Fallback simple summarization using rule-based approach"""
    full_text = transcript_data['full_transcript']
    
    # Split into sentences (simple approach)
    sentences = [s.strip() for s in full_text.replace('\n', ' ').split('.') if s.strip()]
    
    # Score sentences (longer sentences and earlier sentences get higher scores)
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        if len(sentence) > 10:  # Only consider meaningful sentences
            score = len(sentence) * (1 + 0.1 * (len(sentences) - i) / len(sentences))
            scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s[1] for s in scored_sentences[:max_points]]
    
    return {
        'success': True,
        'video_id': transcript_data['video_id'],
        'summary_method': 'simple_rule_based',
        'key_points': top_sentences,
        'statistics': _calculate_statistics(transcript_data)
    }

def _calculate_statistics(transcript_data: Dict) -> Dict:
    """Calculate transcript statistics"""
    full_text = transcript_data['full_transcript']
    word_count = len(full_text.split())
    avg_words_per_minute = word_count / (transcript_data['metadata']['total_duration'] / 60) if transcript_data['metadata']['total_duration'] > 0 else 0
    
    return {
        'total_words': word_count,
        'total_duration_minutes': transcript_data['metadata']['total_duration'] / 60,
        'avg_words_per_minute': avg_words_per_minute,
        'snippet_count': transcript_data['snippet_count'],
        'language': transcript_data['language']
    }

def _extract_key_points_from_text(text: str, max_points: int) -> List[str]:
    """Extract key points from LLM text response when JSON parsing fails"""
    lines = text.split('\n')
    key_points = []
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or 
                     line[0].isdigit() and '.' in line[:3]):
            # Clean up the line
            clean_line = line.lstrip('-•*0123456789. ')
            if clean_line and len(clean_line) > 10:
                key_points.append(clean_line)
                if len(key_points) >= max_points:
                    break
    
    # If we didn't find enough structured points, split by sentences
    if len(key_points) < max_points:
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        key_points.extend(sentences[:max_points - len(key_points)])
    
    return key_points[:max_points]

@app.tool()
def ping() -> str:
    """Simple ping to test if the MCP server is working"""
    return "pong from YouTube MCP Server!"

def main() -> None:
    """Run the MCP server"""
    app.run()

if __name__ == "__main__":
    main()
