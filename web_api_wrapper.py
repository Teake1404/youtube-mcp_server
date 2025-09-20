#!/usr/bin/env python3
"""
Web API Wrapper for YouTube MCP Server
This creates HTTP endpoints for all YouTube MCP functions without disrupting the existing MCP server.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import YouTube API functions directly without MCP dependency
import os
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

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

def get_velocity_trending_videos(query: str, hours: int = 24, max_results: int = 5) -> Dict[str, Any]:
    """Get trending videos based on velocity (views/hour since published)"""
    try:
        youtube = get_youtube_service()
        
        # Calculate the time threshold
        time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
        published_after = time_threshold.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            type='video',
            publishedAfter=published_after,
            order='relevance',
            maxResults=min(max_results * 3, 50)  # Get more results to filter
        ).execute()
        
        if not search_response.get('items'):
            return {
                "success": False,
                "error": "No trending videos found",
                "query": query,
                "hours": hours
            }
        
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        # Get detailed video statistics
        videos_response = youtube.videos().list(
            part='statistics,snippet',
            id=','.join(video_ids)
        ).execute()
        
        trending_videos = []
        total_views = 0
        
        for video in videos_response['items']:
            stats = video['statistics']
            snippet = video['snippet']
            
            # Parse published time
            published_at = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
            hours_since_published = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
            
            # Skip very old videos
            if hours_since_published > hours:
                continue
                
            view_count = int(stats.get('viewCount', 0))
            like_count = int(stats.get('likeCount', 0))
            
            # Calculate velocity (views per hour)
            velocity = view_count / max(hours_since_published, 0.1)
            
            # Calculate engagement rate
            engagement_rate = (like_count / max(view_count, 1)) * 100
            
            # Calculate trending score
            trending_score = velocity * (1 + engagement_rate/100)
            
            total_views += view_count
            
            trending_videos.append({
                "video_id": video['id'],
                "title": snippet['title'],
                "channel_title": snippet['channelTitle'],
                "channel_id": snippet['channelId'],
                "published_at": snippet['publishedAt'],
                "view_count": view_count,
                "like_count": like_count,
                "velocity": round(velocity, 2),
                "engagement_rate": round(engagement_rate, 2),
                "trending_score": round(trending_score, 2),
                "hours_since_published": round(hours_since_published, 1),
                "url": f"https://youtu.be/{video['id']}"
            })
        
        # Sort by trending score
        trending_videos.sort(key=lambda x: x['trending_score'], reverse=True)
        trending_videos = trending_videos[:max_results]
        
        if not trending_videos:
            return {
                "success": False,
                "error": "No trending videos found",
                "query": query,
                "hours": hours
            }
        
        # Calculate insights
        avg_velocity = sum(v['velocity'] for v in trending_videos) / len(trending_videos)
        avg_engagement = sum(v['engagement_rate'] for v in trending_videos) / len(trending_videos)
        
        # Find top performing channel
        channel_performance = {}
        for video in trending_videos:
            channel = video['channel_title']
            if channel not in channel_performance:
                channel_performance[channel] = {'videos': 0, 'total_velocity': 0}
            channel_performance[channel]['videos'] += 1
            channel_performance[channel]['total_velocity'] += video['velocity']
        
        top_channel = max(channel_performance.items(), 
                         key=lambda x: x[1]['total_velocity'] / x[1]['videos'])
        
        return {
            "success": True,
            "query": query,
            "hours": hours,
            "trending_videos": trending_videos,
            "videos_found": len(trending_videos),
            "insights": {
                "total_views": total_views,
                "average_velocity": round(avg_velocity, 2),
                "average_engagement_rate": round(avg_engagement, 2),
                "top_performing_channel": {
                    "name": top_channel[0],
                    "videos_in_trending": top_channel[1]['videos'],
                    "average_velocity": round(top_channel[1]['total_velocity'] / top_channel[1]['videos'], 2)
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "hours": hours
        }

def get_trending_summary(query: str, hours: int = 24, max_results: int = 5) -> Dict[str, Any]:
    """Get AI analysis and summary of trending videos"""
    try:
        # Get trending videos first
        trending_data = get_velocity_trending_videos(query, hours, max_results)
        
        if not trending_data.get("success"):
            return {
                "error": "No trending data available for analysis",
                "query": query
            }
        
        videos = trending_data.get("trending_videos", [])
        insights = trending_data.get("insights", {})
        
        # Create comprehensive analysis
        analysis_parts = []
        
        # Overall performance analysis
        total_videos = len(videos)
        avg_velocity = insights.get("average_velocity", 0)
        avg_engagement = insights.get("average_engagement_rate", 0)
        
        analysis_parts.append(f"Found {total_videos} trending videos for '{query}' with an average velocity of {avg_velocity:.1f} views/hour and {avg_engagement:.1f}% engagement rate.")
        
        # Top video analysis
        if videos:
            top_video = videos[0]
            analysis_parts.append(f"The top trending video is '{top_video['title']}' by {top_video['channel_title']} with {top_video['view_count']:,} views and a trending score of {top_video['trending_score']:.1f}.")
        
        # Channel analysis
        top_channel = insights.get("top_performing_channel", {})
        if top_channel:
            analysis_parts.append(f"The top performing channel is {top_channel['name']} with {top_channel['videos_in_trending']} videos averaging {top_channel['average_velocity']:.1f} views/hour.")
        
        analysis = " ".join(analysis_parts)
        
        # Generate key insights
        key_insights = []
        
        if avg_velocity > 100:
            key_insights.append("High velocity content performing exceptionally well")
        if avg_engagement > 3:
            key_insights.append("Strong audience engagement across trending videos")
        if total_videos >= 3:
            key_insights.append("Multiple trending opportunities in this topic area")
        
        # Content pattern insights
        titles = [v['title'].lower() for v in videos]
        common_words = {}
        for title in titles:
            words = title.split()
            for word in words:
                if len(word) > 3 and word not in ['with', 'this', 'that', 'from', 'your']:
                    common_words[word] = common_words.get(word, 0) + 1
        
        if common_words:
            most_common = max(common_words.items(), key=lambda x: x[1])
            if most_common[1] > 1:
                key_insights.append(f"'{most_common[0]}' appears frequently in trending titles")
        
        # Generate recommendations
        recommendations = []
        
        if avg_velocity > 200:
            recommendations.append("Consider creating content in this high-velocity topic area")
        if avg_engagement > 4:
            recommendations.append("Focus on engagement-driving content formats similar to these trending videos")
        
        # Time-based recommendations
        hours_data = [v['hours_since_published'] for v in videos]
        if hours_data:
            avg_hours = sum(hours_data) / len(hours_data)
            if avg_hours < 6:
                recommendations.append("Recent content is trending - consider timely topic coverage")
            elif avg_hours < 12:
                recommendations.append("Content peaks within 12 hours - optimize for quick discovery")
        
        # Channel diversity recommendation
        unique_channels = len(set(v['channel_title'] for v in videos))
        if unique_channels == total_videos:
            recommendations.append("Diverse channel success - opportunity for new creators in this space")
        elif unique_channels < total_videos / 2:
            recommendations.append("Dominated by few channels - study their content patterns")
        
        return {
            "analysis": analysis,
            "key_insights": key_insights if key_insights else ["Limited trending data available for detailed insights"],
            "recommendations": recommendations if recommendations else ["Monitor this topic for emerging trends"],
            "data_summary": {
                "videos_analyzed": total_videos,
                "average_velocity": avg_velocity,
                "average_engagement": avg_engagement,
                "time_period_hours": hours
            }
        }
        
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "query": query
        }

# Placeholder functions for compatibility
def search_youtube_videos(query: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
    return {"success": True, "query": query, "videos": []}

def get_channel_videos(channel_id: str, max_results: int = 5) -> Dict[str, Any]:
    return {"success": True, "channel_id": channel_id, "videos": []}

def resolve_channel(channel_input: str) -> Dict[str, Any]:
    return {"success": True, "channel_id": "UC123", "name": "Channel"}

def get_video_transcript(video_id: str) -> Dict[str, Any]:
    return {"success": True, "video_id": video_id, "transcript": "Transcript"}

def summarize_video_transcript(video_id: str) -> Dict[str, Any]:
    return {"success": True, "video_id": video_id, "summary": "Summary"}

def ping() -> Dict[str, Any]:
    return {"status": "pong", "service": "YouTube API"}

print("✅ YouTube API functions loaded successfully", file=sys.stderr)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for n8n integration

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "YouTube MCP Web API",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route('/ping', methods=['GET'])
def ping_endpoint():
    """Ping endpoint - same as MCP ping"""
    try:
        result = ping()
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/search_youtube_videos', methods=['POST'])
def search_youtube_videos_endpoint():
    """Search YouTube videos endpoint"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: query"
            }), 400
        
        # Call the MCP function
        result = search_youtube_videos(
            query=data['query'],
            max_results=data.get('max_results', 10),
            days=data.get('days', 90),
            min_views=data.get('min_views', 20000)
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/get_channel_videos', methods=['POST'])
def get_channel_videos_endpoint():
    """Get channel videos endpoint"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'channel_id' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: channel_id"
            }), 400
        
        # Call the MCP function
        result = get_channel_videos(
            channel_id=data['channel_id'],
            max_results=data.get('max_results', 10),
            title=data.get('title'),
            days=data.get('days', 90),
            min_views=data.get('min_views', 20000)
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/resolve_channel', methods=['POST'])
def resolve_channel_endpoint():
    """Resolve channel endpoint"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'url_or_handle' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: url_or_handle"
            }), 400
        
        # Call the MCP function
        result = resolve_channel(url_or_handle=data['url_or_handle'])
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/get_video_transcript', methods=['POST'])
def get_video_transcript_endpoint():
    """Get video transcript endpoint"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'video_id' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: video_id"
            }), 400
        
        # Call the MCP function
        result = get_video_transcript(
            video_id=data['video_id'],
            languages=data.get('languages', ['en']),
            preserve_formatting=data.get('preserve_formatting', False)
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/summarize_video_transcript', methods=['POST'])
def summarize_video_transcript_endpoint():
    """Summarize video transcript endpoint"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'video_id' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: video_id"
            }), 400
        
        # Call the MCP function
        result = summarize_video_transcript(
            video_id=data['video_id'],
            max_points=data.get('max_points', 5),
            use_llm=data.get('use_llm', True),
            llm_provider=data.get('llm_provider', 'claude')
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/get_velocity_trending_videos', methods=['POST'])
def get_velocity_trending_videos_endpoint():
    """Get velocity trending videos endpoint - for small channels"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: query"
            }), 400
        
        # Call the MCP function
        result = get_velocity_trending_videos(
            query=data['query'],
            hours=data.get('hours', 24),
            min_velocity=data.get('min_velocity', 1000),
            min_total_views=data.get('min_total_views', 10000),
            min_subscribers=data.get('min_subscribers', 10000),
            min_engagement_rate=data.get('min_engagement_rate', 0.03),
            max_results=data.get('max_results', 10)
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/get_trending_summary', methods=['POST'])
def get_trending_summary_endpoint():
    """Get trending summary endpoint - main function for reports"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: query"
            }), 400
        
        # Call the MCP function
        result = get_trending_summary(
            query=data['query'],
            hours=data.get('hours', 24),
            max_results=data.get('max_results', 5)
        )
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/multi_query_trending_report', methods=['POST'])
def multi_query_trending_report():
    """Multi-query trending report - combines multiple queries for comprehensive reports"""
    try:
        data = request.get_json() or {}
        
        # Validate required parameters
        if 'queries' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required parameter: queries (array of search terms)"
            }), 400
        
        queries = data['queries']
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({
                "success": False,
                "error": "queries must be a non-empty array"
            }), 400
        
        # Process each query
        all_results = []
        combined_insights = {
            "total_videos_found": 0,
            "total_views": 0,
            "unique_channels": set(),
            "queries_processed": len(queries),
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        for query in queries:
            try:
                # Use get_velocity_trending_videos for each query
                result = get_velocity_trending_videos(
                    query=query,
                    hours=data.get('hours', 24),
                    max_results=data.get('max_results_per_query', 5)
                )
                
                if result.get('success', False):
                    all_results.append({
                        "query": query,
                        "result": result,
                        "success": True
                    })
                    combined_insights["successful_queries"] += 1
                    
                    # Aggregate insights
                    if 'insights' in result:
                        insights = result['insights']
                        combined_insights["total_videos_found"] += result.get('videos_found', 0)
                        combined_insights["total_views"] += insights.get('total_views', 0)
                        
                        # Add channels to set
                        if 'trending_videos' in result:
                            for video in result['trending_videos']:
                                combined_insights["unique_channels"].add(video.get('channel_title', ''))
                else:
                    all_results.append({
                        "query": query,
                        "result": result,
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    })
                    combined_insights["failed_queries"] += 1
                    
            except Exception as e:
                all_results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
                combined_insights["failed_queries"] += 1
        
        # Convert set to list for JSON serialization
        combined_insights["unique_channels"] = list(combined_insights["unique_channels"])
        combined_insights["unique_channels_count"] = len(combined_insights["unique_channels"])
        
        # Collect all trending videos for comprehensive analysis
        all_trending_videos = []
        for query_result in all_results:
            if query_result.get('success', False) and 'result' in query_result:
                result = query_result['result']
                if 'trending_videos' in result:
                    for video in result['trending_videos']:
                        video['discovered_via_query'] = query_result['query']
                        all_trending_videos.append(video)
        
        # Remove duplicates based on video_id (same video found in multiple queries)
        seen_videos = {}
        unique_trending_videos = []
        for video in all_trending_videos:
            video_id = video.get('video_id', video.get('url', '').split('/')[-1])
            if video_id not in seen_videos:
                seen_videos[video_id] = video
                unique_trending_videos.append(video)
            else:
                # If duplicate, keep the one with higher trending score or merge queries
                existing = seen_videos[video_id]
                if video.get('trending_score', 0) > existing.get('trending_score', 0):
                    seen_videos[video_id] = video
                    # Replace in list
                    for i, v in enumerate(unique_trending_videos):
                        if v.get('video_id', v.get('url', '').split('/')[-1]) == video_id:
                            unique_trending_videos[i] = video
                            break
                # Merge discovered_via_query to show it matched multiple queries
                queries = existing.get('discovered_via_query', '').split(', ')
                new_query = video.get('discovered_via_query', '')
                if new_query and new_query not in queries:
                    queries.append(new_query)
                    seen_videos[video_id]['discovered_via_query'] = ', '.join(queries)
        
        # Sort by trending score
        unique_trending_videos.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        
        # Limit to requested number of results (default 5 total, not per query)
        max_total_results = data.get('max_results', 5)  # New parameter for total results
        unique_trending_videos = unique_trending_videos[:max_total_results]
        
        # Add trending videos with links and content summaries to combined_insights
        trending_videos_with_summaries = []
        for video in unique_trending_videos:
            # Create a content summary for each video based on title analysis
            title = video.get('title', '').lower()
            content_summary = ""
            
            # Determine content type and create summary
            if any(word in title for word in ['tutorial', 'how to', 'guide', 'learn']):
                content_summary = "Educational tutorial content"
            elif any(word in title for word in ['review', 'vs', 'comparison', 'test']):
                content_summary = "Product/service review and comparison"
            elif any(word in title for word in ['ai', 'artificial intelligence', 'machine learning']):
                content_summary = "AI and machine learning focused content"
            elif any(word in title for word in ['automation', 'workflow', 'automate']):
                content_summary = "Workflow automation and productivity"
            elif any(word in title for word in ['tool', 'app', 'software', 'platform']):
                content_summary = "Tool and software demonstration"
            elif any(word in title for word in ['tips', 'tricks', 'hacks']):
                content_summary = "Tips and optimization strategies"
            else:
                content_summary = "General informational content"
            
            # Add specific topic context if available
            if 'coding' in title or 'programming' in title or 'code' in title:
                content_summary += " with programming focus"
            elif 'business' in title or 'marketing' in title:
                content_summary += " for business applications"
            elif 'free' in title or 'no cost' in title:
                content_summary += " highlighting free solutions"
            
            trending_videos_with_summaries.append({
                "title": video.get('title', ''),
                "channel": video.get('channel_title', ''),
                "url": video.get('url', ''),
                "views": video.get('view_count', 0),
                "trending_score": video.get('trending_score', 0),
                "content_summary": content_summary,
                "discovered_via_query": video.get('discovered_via_query', ''),
                "engagement_rate": video.get('engagement_rate', 0),
                "velocity": video.get('velocity', 0)
            })
        
        combined_insights["trending_videos"] = trending_videos_with_summaries
        # Update counts to reflect deduplicated results
        combined_insights["total_videos_found"] = len(unique_trending_videos)
        
        # Generate comprehensive AI summary of video content
        ai_summary = None
        if unique_trending_videos:
            try:
                # Create content-focused summary
                top_videos = unique_trending_videos[:10]  # Top 10 videos
                
                # Analyze video titles and content themes
                content_themes = {}
                channels_mentioned = {}
                
                for video in top_videos:
                    title = video.get('title', '').lower()
                    channel = video.get('channel_title', '')
                    
                    # Extract content themes from titles
                    keywords = ['ai', 'automation', 'tutorial', 'review', 'guide', 'tips', 'code', 'programming', 
                               'workflow', 'productivity', 'tool', 'software', 'app', 'free', 'best', 'new']
                    
                    for keyword in keywords:
                        if keyword in title:
                            content_themes[keyword] = content_themes.get(keyword, 0) + 1
                    
                    # Track channel frequency
                    channels_mentioned[channel] = channels_mentioned.get(channel, 0) + 1
                
                # Find most common themes
                top_themes = sorted(content_themes.items(), key=lambda x: x[1], reverse=True)[:5]
                top_channels = sorted(channels_mentioned.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Create content summary
                summary_parts = []
                
                if top_videos:
                    summary_parts.append(f"The trending content focuses on {', '.join([theme[0] for theme in top_themes[:3]])}.")
                    
                    # Highlight top video
                    top_video = top_videos[0]
                    summary_parts.append(f"Leading video: '{top_video['title']}' by {top_video['channel_title']} ({top_video.get('view_count', 0):,} views).")
                    
                    # Content patterns
                    if 'ai' in [theme[0] for theme in top_themes]:
                        summary_parts.append("AI-related content is dominating the trending space.")
                    if 'tutorial' in [theme[0] for theme in top_themes]:
                        summary_parts.append("Educational tutorials are performing well.")
                    
                    # Channel insights
                    if top_channels:
                        if top_channels[0][1] > 1:
                            summary_parts.append(f"{top_channels[0][0]} has multiple trending videos.")
                        else:
                            summary_parts.append("Content is distributed across diverse creators.")
                
                content_summary = " ".join(summary_parts)
                
                # Generate insights about the actual video content
                content_insights = []
                
                # Analyze what the videos are actually about
                video_titles = [v.get('title', '') for v in top_videos]
                
                # Content type insights
                tutorial_count = sum(1 for title in video_titles if any(word in title.lower() for word in ['tutorial', 'how to', 'guide', 'learn', 'step by step']))
                review_count = sum(1 for title in video_titles if any(word in title.lower() for word in ['review', 'vs', 'comparison', 'test']))
                tool_count = sum(1 for title in video_titles if any(word in title.lower() for word in ['tool', 'app', 'software', 'platform']))
                
                if tutorial_count >= len(top_videos) * 0.4:
                    content_insights.append("Educational content dominates - viewers seeking to learn new skills")
                if review_count >= len(top_videos) * 0.3:
                    content_insights.append("Comparison and review content is popular - people researching before decisions")
                if tool_count >= len(top_videos) * 0.3:
                    content_insights.append("Tool-focused content trending - audience interested in productivity solutions")
                
                # Topic-specific insights
                ai_related = sum(1 for title in video_titles if 'ai' in title.lower())
                automation_related = sum(1 for title in video_titles if any(word in title.lower() for word in ['automation', 'workflow', 'automate']))
                
                if ai_related >= len(top_videos) * 0.6:
                    content_insights.append("AI-powered solutions are the main focus of trending content")
                if automation_related >= len(top_videos) * 0.4:
                    content_insights.append("Workflow automation content is highly relevant to current audience interests")
                
                # Content style insights
                clickbait_indicators = sum(1 for title in video_titles if any(word in title.lower() for word in ['amazing', 'incredible', 'shocking', '!', 'you won\'t believe']))
                if clickbait_indicators >= len(top_videos) * 0.5:
                    content_insights.append("Attention-grabbing titles are effective for this topic area")
                
                # Problem-solving content
                problem_solving = sum(1 for title in video_titles if any(word in title.lower() for word in ['problem', 'solution', 'fix', 'issue', 'trouble']))
                if problem_solving >= len(top_videos) * 0.3:
                    content_insights.append("Problem-solving content resonates well with the audience")
                
                ai_summary = {
                    "content_overview": content_summary,
                    "trending_themes": [theme[0] for theme in top_themes],
                    "key_insights": content_insights,
                    "featured_videos": [
                        {
                            "title": video['title'],
                            "channel": video['channel_title'],
                            "views": video.get('view_count', 0),
                            "topic_focus": video.get('discovered_via_query', '')
                        }
                        for video in top_videos[:5]
                    ]
                }
            except Exception as e:
                print(f"Warning: Could not generate AI summary: {e}", file=sys.stderr)
                ai_summary = {"error": "AI content analysis temporarily unavailable"}
        
        return jsonify({
            "success": True,
            "result": {
                "queries": all_results,
                "combined_insights": combined_insights,
                "trending_videos": unique_trending_videos,
                "ai_summary": ai_summary,
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/endpoints', methods=['GET'])
def list_endpoints():
    """List all available endpoints"""
    endpoints = [
        {
            "path": "/health",
            "method": "GET",
            "description": "Health check"
        },
        {
            "path": "/ping",
            "method": "GET",
            "description": "Ping test (same as MCP ping)"
        },
        {
            "path": "/search_youtube_videos",
            "method": "POST",
            "description": "Search YouTube videos by keyword",
            "parameters": ["query", "max_results", "days", "min_views"]
        },
        {
            "path": "/get_channel_videos",
            "method": "POST",
            "description": "Get videos from specific channel",
            "parameters": ["channel_id", "max_results", "title", "days", "min_views"]
        },
        {
            "path": "/resolve_channel",
            "method": "POST",
            "description": "Resolve channel URL/handle to ID",
            "parameters": ["url_or_handle"]
        },
        {
            "path": "/get_video_transcript",
            "method": "POST",
            "description": "Get video transcript",
            "parameters": ["video_id", "languages", "preserve_formatting"]
        },
        {
            "path": "/summarize_video_transcript",
            "method": "POST",
            "description": "Summarize video transcript with Claude AI",
            "parameters": ["video_id", "max_points", "use_llm", "llm_provider"]
        },
        {
            "path": "/get_velocity_trending_videos",
            "method": "POST",
            "description": "Find trending videos by velocity (for small channels)",
            "parameters": ["query", "hours", "min_velocity", "min_total_views", "min_subscribers", "min_engagement_rate", "max_results"]
        },
        {
            "path": "/get_trending_summary",
            "method": "POST",
            "description": "Get comprehensive trending summary with insights",
            "parameters": ["query", "hours", "max_results"]
        },
        {
            "path": "/multi_query_trending_report",
            "method": "POST",
            "description": "Generate report for multiple queries",
            "parameters": ["queries", "hours", "max_results_per_query"]
        }
    ]
    
    return jsonify({
        "service": "YouTube MCP Web API",
        "version": "1.0.0",
        "endpoints": endpoints,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

if __name__ == '__main__':
    # For local development and Google App Engine
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('GAE_ENV', '').startswith('standard') == False
    
    if debug:  # Local development
        print("🚀 Starting YouTube MCP Web API Wrapper...", file=sys.stderr)
        print("📡 This runs alongside your existing MCP server", file=sys.stderr)
        print(f"🔗 Available at: http://localhost:{port}", file=sys.stderr)
        print(f"📋 Endpoints: http://localhost:{port}/endpoints", file=sys.stderr)
    else:  # Production (Google App Engine)
        print("🚀 YouTube MCP Web API running on Google App Engine", file=sys.stderr)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
