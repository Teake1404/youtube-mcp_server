#!/usr/bin/env python3
"""
YouTube API Test Script - Code along with this!
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()


def build_service():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("❌ No YouTube API key found!")
        print("📝 Steps:")
        print("   1. Enable YouTube Data API v3 in Google Cloud Console")
        print("   2. Create an API key")
        print("   3. Add YOUTUBE_API_KEY=your_key_here to .env")
        sys.exit(1)
    return build('youtube', 'v3', developerKey=api_key)


def resolve_channel_id(youtube, url_or_handle: str) -> str:
    """Resolve a channel URL or @handle to a UC... channel ID."""
    text = url_or_handle.strip()
    # Extract handle from common URL formats
    # Examples: https://www.youtube.com/@Handle, https://youtube.com/@Handle
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
        raise SystemExit(f"Could not resolve channel from: {url_or_handle}")
    return items[0]['snippet']['channelId']


def _get_video_stats(youtube, video_ids: List[str]) -> Dict[str, Dict]:
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


def search_by_niche(youtube, query: str, max_results: int = 10, days: int | None = None, min_views: int | None = None) -> List[Dict]:
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
    stats = _get_video_stats(youtube, ids)

    candidates: List[Tuple[int, Dict]] = []
    for vid, meta in stats.items():
        views = meta['view_count'] or 0
        if min_views is not None and views < min_views:
            continue
        candidates.append((views, {
            'video_id': vid,
            'title': meta['title'],
            'channel_title': meta['channel_title'],
            'published_at': meta['published_at'],
        }))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in candidates[:max_results]]


def videos_by_channel(youtube, channel_id: str, max_results: int = 10, title: str | None = None, days: int | None = None, min_views: int | None = None) -> List[Dict]:
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
    stats = _get_video_stats(youtube, ids)

    candidates: List[Tuple[int, Dict]] = []
    for vid, meta in stats.items():
        views = meta['view_count'] or 0
        if min_views is not None and views < min_views:
            continue
        candidates.append((views, {
            'video_id': vid,
            'title': meta['title'],
            'channel_title': meta['channel_title'],
            'published_at': meta['published_at'],
        }))

    # If title search was used, we'll already have 1 best match; otherwise, sort by views desc to get popular and recent
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in candidates[:max_results]]


def get_video_transcript(video_id: str, languages: List[str] = None, preserve_formatting: bool = False) -> Dict:
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


def summarize_transcript(transcript_data: Dict, max_points: int = 5, use_llm: bool = True, llm_provider: str = "claude") -> Dict:
    """
    Create a sophisticated summary of the transcript using LLM capabilities.
    
    Args:
        transcript_data: Transcript data from get_video_transcript
        max_points: Maximum number of summary points to generate
        use_llm: Whether to use LLM for summarization (default: True)
        llm_provider: LLM provider to use ('claude', 'openai', or 'simple')
    
    Returns:
        Dict containing summary information
    """
    if not transcript_data.get('success'):
        return {
            'success': False,
            'error': 'No transcript data available for summarization'
        }
    
    full_text = transcript_data['full_transcript']
    raw_snippets = transcript_data['raw_snippets']
    
    # If LLM is disabled or failed, fall back to simple summarization
    if not use_llm or llm_provider == "simple":
        return _simple_summarization(transcript_data, max_points)
    
    try:
        if llm_provider == "claude":
            return _claude_summarization(transcript_data, max_points)
        else:
            print(f"⚠️  Unknown LLM provider '{llm_provider}', falling back to simple summarization")
            return _simple_summarization(transcript_data, max_points)
            
    except Exception as e:
        print(f"⚠️  LLM summarization failed: {e}, falling back to simple summarization")
        return _simple_summarization(transcript_data, max_points)


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
            model="claude-sonnet-4-20250514",
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
    raw_snippets = transcript_data['raw_snippets']
    
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


def main():
    parser = argparse.ArgumentParser(description='YouTube API utilities')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('niche', help='Search by niche keyword(s) with popularity/recency filters')
    p1.add_argument('query', help='Keyword(s), e.g. "ai coding"')
    p1.add_argument('--max', type=int, default=10, help='Max results')
    p1.add_argument('--days', type=int, default=90, help='Only include videos published in the last N days (default: 90)')
    p1.add_argument('--min-views', type=int, default=20000, help='Only include videos with at least this many views (default: 20000)')

    p2 = sub.add_parser('channel', help='List recent videos or find a specific video in a channel, with popularity filters')
    p2.add_argument('channel_id', help='Channel ID (starts with UC...)')
    p2.add_argument('--max', type=int, default=10, help='Max results')
    p2.add_argument('--title', type=str, help='Specific video title to find within the channel')
    p2.add_argument('--days', type=int, default=90, help='Only include videos published in the last N days (default: 90)')
    p2.add_argument('--min-views', type=int, default=20000, help='Only include videos with at least this many views (default: 20000)')

    p3 = sub.add_parser('resolve', help='Resolve a channel URL or @handle to channel ID')
    p3.add_argument('url_or_handle', help='Channel URL or @handle')

    # Add transcript-related commands
    p4 = sub.add_parser('transcript', help='Get transcript for a specific video')
    p4.add_argument('video_id', help='YouTube video ID')
    p4.add_argument('--languages', nargs='+', default=['en'], help='Language codes to try (default: en)')
    p4.add_argument('--preserve-formatting', action='store_true', help='Preserve HTML formatting in transcript')
    p4.add_argument('--summary', action='store_true', help='Generate summary of transcript')
    p4.add_argument('--max-points', type=int, default=5, help='Maximum summary points (default: 5)')
    p4.add_argument('--llm', choices=['claude', 'simple'], default='claude', help='LLM provider for summarization (default: claude)')

    p5 = sub.add_parser('transcript-batch', help='Get transcripts for multiple videos from search results')
    p5.add_argument('query', help='Search query for videos')
    p5.add_argument('--max', type=int, default=3, help='Max videos to process (default: 3)')
    p5.add_argument('--days', type=int, default=90, help='Only include videos published in the last N days (default: 90)')
    p5.add_argument('--min-views', type=int, default=20000, help='Only include videos with at least this many views (default: 20000)')
    p5.add_argument('--languages', nargs='+', default=['en'], help='Language codes to try (default: en)')
    p5.add_argument('--summary', action='store_true', help='Generate summary for each transcript')
    p5.add_argument('--max-points', type=int, default=5, help='Maximum summary points (default: 5)')
    p5.add_argument('--llm', choices=['claude', 'simple'], default='claude', help='LLM provider for summarization (default: claude)')

    args = parser.parse_args()
    youtube = build_service()

    if args.cmd == 'niche':
        results = search_by_niche(youtube, args.query, args.max, args.days, args.min_views)
    elif args.cmd == 'channel':
        results = videos_by_channel(youtube, args.channel_id, args.max, args.title, args.days, args.min_views)
    elif args.cmd == 'transcript':
        # Handle single video transcript
        transcript_data = get_video_transcript(args.video_id, args.languages, args.preserve_formatting)
        
        if transcript_data['success']:
            print(f"✅ Transcript retrieved for video: {args.video_id}")
            print(f"📝 Language: {transcript_data['language']} ({transcript_data['language_code']})")
            print(f"🔧 Generated: {'Yes' if transcript_data['is_generated'] else 'No'}")
            print(f"📊 Snippets: {transcript_data['snippet_count']}")
            print(f"⏱️  Duration: {transcript_data['metadata']['total_duration']:.1f}s")
            print(f"📖 Word count: {len(transcript_data['full_transcript'].split())}")
            
            if args.summary:
                summary = summarize_transcript(transcript_data, args.max_points, True, args.llm)
                if summary['success']:
                    print(f"\n📋 SUMMARY:")
                    print(f"   Method: {summary['summary_method']}")
                    if summary.get('executive_summary'):
                        print(f"   • Executive Summary: {summary['executive_summary']}")
                    print(f"   • Key Points:")
                    for i, point in enumerate(summary['key_points'], 1):
                        print(f"      {i}. {point}")
                    if summary.get('tone_style'):
                        print(f"   • Tone: {summary['tone_style']}")
                    if summary.get('target_audience'):
                        print(f"   • Target Audience: {summary['target_audience']}")
                    print(f"\n📈 Statistics:")
                    stats = summary['statistics']
                    print(f"   • Total words: {stats['total_words']}")
                    print(f"   • Duration: {stats['total_duration_minutes']:.1f} minutes")
                    print(f"   • Words per minute: {stats['avg_words_per_minute']:.1f}")
                else:
                    print(f"❌ Summary failed: {summary['error']}")
            
            if not args.summary:
                print(f"\n📄 FULL TRANSCRIPT:")
                print(transcript_data['full_transcript'][:500] + "..." if len(transcript_data['full_transcript']) > 500 else transcript_data['full_transcript'])
        else:
            print(f"❌ Failed to get transcript: {transcript_data['error']}")
        return
        
    elif args.cmd == 'transcript-batch':
        # Handle batch transcript processing
        print(f"🔍 Searching for videos matching: {args.query}")
        results = search_by_niche(youtube, args.query, args.max, args.days, args.min_views)
        
        if not results:
            print("❌ No videos found matching criteria")
            return
        
        print(f"\n📹 Found {len(results)} videos. Processing transcripts...\n")
        
        for i, video in enumerate(results, 1):
            video_id = video['video_id']
            print(f"🎬 {i}/{len(results)}: {video['title']}")
            print(f"   Channel: {video['channel_title']}")
            print(f"   URL: https://youtu.be/{video_id}")
            
            transcript_data = get_video_transcript(video_id, args.languages)
            
            if transcript_data['success']:
                print(f"   ✅ Transcript: {transcript_data['language']} ({len(transcript_data['full_transcript'].split())} words)")
                
                if args.summary:
                    max_points = getattr(args, 'max_points', 5)  # Default to 5 if not specified
                    summary = summarize_transcript(transcript_data, max_points, True, args.llm)
                    if summary['success']:
                        print(f"   📋 Key points:")
                        for j, point in enumerate(summary['key_points'][:3], 1):  # Show top 3 points
                            print(f"      {j}. {point[:80]}{'...' if len(point) > 80 else ''}")
            else:
                print(f"   ❌ Transcript failed: {transcript_data['error']}")
            
            print()  # Empty line between videos
        return
    else:
        cid = resolve_channel_id(youtube, args.url_or_handle)
        print(cid)
        return

    # Display results for original commands
    for i, r in enumerate(results, 1):
        print(f"{i:02d}. {r['title']} | {r['channel_title']} | {r['published_at']} | https://youtu.be/{r['video_id']}")


if __name__ == "__main__":
    main()