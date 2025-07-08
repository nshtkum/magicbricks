import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import requests
from datetime import datetime
import time
from typing import List, Dict, Any
from urllib.parse import urlparse
import urllib.robotparser
from bs4 import BeautifulSoup
import hashlib

# App config
st.set_page_config(page_title="Qforia Pro Enhanced", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .data-point {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .numerical-highlight {
        background: #fff2e6;
        color: #cc6600;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }
    .enhancement-suggestion {
        background: #f0fff0;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .missing-topic {
        background: #fff5f5;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .content-analysis {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Qforia Pro Enhanced: Query Fan-Out & Content Analysis Tool</h1>
    <p>AI-Powered Query Expansion, Real-Time Fact Verification & Content Enhancement</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = []
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}
if 'content_analysis' not in st.session_state:
    st.session_state.content_analysis = None
if 'enhancement_suggestions' not in st.session_state:
    st.session_state.enhancement_suggestions = None

# Sidebar
st.sidebar.header("🔧 Configuration")

# API Keys
try:
    gemini_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
    perplexity_key = st.secrets["api_keys"]["PERPLEXITY_API_KEY"]
    st.sidebar.success("🔑 API Keys loaded from secrets")
except:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password", help="Enter your Perplexity API key")

# Validate API keys
if not perplexity_key or not perplexity_key.startswith('pplx-'):
    st.sidebar.warning("⚠️ Valid Perplexity API key required (starts with 'pplx-')")
if not gemini_key:
    st.sidebar.warning("⚠️ Gemini API key required")

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Enhanced URL scraping function with fallback
def scrape_url_content(url, use_fallback=False):
    """Scrape content from URL with multiple fallback methods"""
    if use_fallback:
        return None, "Please paste content manually"
    
    try:
        # Check robots.txt first
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Try to respect robots.txt
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            rp.read()
            if not rp.can_fetch('*', url):
                return None, "Access blocked by robots.txt"
        except:
            pass  # Continue if robots.txt check fails
        
        # Multiple request strategies
        strategies = [
            # Strategy 1: Standard request
            lambda: requests.get(url, headers=headers, timeout=15),
            # Strategy 2: Session with cookies
            lambda: requests.Session().get(url, headers=headers, timeout=15),
            # Strategy 3: Different user agent
            lambda: requests.get(url, headers={**headers, 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}, timeout=15)
        ]
        
        response = None
        for i, strategy in enumerate(strategies):
            try:
                response = strategy()
                if response.status_code == 200:
                    break
                time.sleep(1)  # Brief delay between attempts
            except Exception as e:
                if i == len(strategies) - 1:  # Last strategy
                    return None, f"All request strategies failed: {str(e)}"
                continue
        
        if not response or response.status_code != 200:
            return None, f"HTTP Error: {response.status_code if response else 'No response'}"
        
        # Parse content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to extract main content
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content', 
            '.entry-content', '.article-content', '.post-body', '.content-body'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
        
        # Extract metadata
        metadata = {
            'title': soup.title.string if soup.title else 'No title found',
            'word_count': len(text.split()),
            'char_count': len(text),
            'url': url,
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {'content': text, 'metadata': metadata}, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Parsing error: {str(e)}"

# API Functions (keeping existing ones and adding new)
def call_perplexity_api(query):
    """Call Perplexity API for research"""
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant. Provide detailed, factual information with specific numbers and statistics where available."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            st.session_state.api_usage['perplexity_calls'] += 1
            return response.json()
        elif response.status_code == 400:
            alternative_models = [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online"
            ]
            
            for model in alternative_models:
                simple_data = {
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                
                simple_response = requests.post("https://api.perplexity.ai/chat/completions", 
                                              headers=headers, json=simple_data, timeout=30)
                
                if simple_response.status_code == 200:
                    st.session_state.api_usage['perplexity_calls'] += 1
                    return simple_response.json()
            
            error_details = response.text if response.text else "No error details"
            return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
        else:
            error_details = response.text if response.text else "No error details"
            return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
    
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def call_perplexity_answer_api(query):
    """Call Perplexity Answer API for fact-checking"""
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "user", "content": f"Please provide a factual answer with sources for: {query}"}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            st.session_state.api_usage['perplexity_calls'] += 1
            result = response.json()
            
            if 'choices' in result and result['choices']:
                answer = result['choices'][0]['message']['content']
                return {"answer": answer, "sources": []}
            else:
                return {"error": "No answer received"}
        else:
            error_details = response.text if response.text else "No error details"
            return {"error": f"API call failed with status {response.status_code}. Details: {error_details}"}
    
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def extract_data_points(text):
    """Extract numerical data points from text"""
    data_points = []
    
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        patterns = [
            (r'(\d+(?:\.\d+)?%)', 'Percentage'),
            (r'(\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)', 'Currency'),
            (r'(₹\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)', 'Currency'),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|crore|lakh))', 'Large Number'),
            (r'(\d{4}(?:-\d{4})?)', 'Year'),
            (r'(\d+(?:\.\d+)?\s*(?:sq\s*ft|acres|hectares|sqft))', 'Area'),
            (r'(\d+(?:\.\d+)?\s*(?:years?|months?|days?))', 'Time Period'),
        ]
        
        for pattern, data_type in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                context = sentence.replace('\n', ' ').strip()
                if len(context) > 150:
                    match_pos = context.find(match)
                    start = max(0, match_pos - 75)
                    end = min(len(context), match_pos + 75)
                    context = context[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(sentence):
                        context = context + "..."
                
                data_points.append({
                    'value': match,
                    'type': data_type,
                    'description': context
                })
    
    seen = set()
    unique_data_points = []
    for dp in data_points:
        identifier = (dp['value'], dp['type'])
        if identifier not in seen:
            seen.add(identifier)
            unique_data_points.append(dp)
    
    return unique_data_points

# Content Analysis Functions
def analyze_content_structure(content):
    """Analyze content structure and extract key topics"""
    try:
        prompt = f"""
        Analyze this article content and provide a detailed structural analysis:

        Content: {content[:4000]}...

        Provide analysis in this JSON format:
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "content_type": "blog_post|news_article|guide|analysis",
            "tone": "professional|casual|technical|marketing",
            "target_audience": "general|professionals|investors|students",
            "key_sections": ["section1", "section2"],
            "writing_style": "descriptive analysis",
            "content_gaps": ["missing_topic1", "missing_topic2"],
            "strengths": ["strength1", "strength2"],
            "word_count_estimate": 1000,
            "readability_level": "intermediate|beginner|advanced"
        }}
        """
        
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        analysis = json.loads(json_text)
        return analysis
        
    except Exception as e:
        st.error(f"Error analyzing content: {e}")
        return None

def generate_enhancement_suggestions(content, analysis):
    """Generate specific enhancement suggestions for the content"""
    try:
        prompt = f"""
        Based on this content analysis, provide specific enhancement suggestions:

        Content Analysis: {json.dumps(analysis, indent=2)}
        Content Sample: {content[:2000]}...

        Provide suggestions in this JSON format:
        {{
            "missing_data_points": [
                {{
                    "category": "Statistics",
                    "suggestion": "Add current market statistics",
                    "example": "Include 2024 growth percentages",
                    "priority": "high|medium|low",
                    "reasoning": "Why this is important"
                }}
            ],
            "topic_expansions": [
                {{
                    "topic": "Topic to expand",
                    "current_coverage": "brief|moderate|detailed",
                    "suggested_additions": ["addition1", "addition2"],
                    "potential_subtopics": ["subtopic1", "subtopic2"]
                }}
            ],
            "content_improvements": [
                {{
                    "area": "Structure|Data|Examples|Sources",
                    "improvement": "Specific improvement",
                    "implementation": "How to implement",
                    "impact": "Expected impact"
                }}
            ],
            "seo_enhancements": [
                {{
                    "type": "Keywords|Headers|Meta",
                    "suggestion": "Specific SEO suggestion",
                    "implementation": "How to implement"
                }}
            ],
            "fact_check_needed": [
                {{
                    "claim": "Claim that needs verification",
                    "reason": "Why it needs checking",
                    "priority": "high|medium|low"
                }}
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        suggestions = json.loads(json_text)
        return suggestions
        
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return None

# Query Fan-Out Functions (keeping existing)
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        target = min_queries_simple
        instruction = f"Generate {min_queries_simple}-{min_queries_simple + 2} queries for a simple overview"
    else:
        target = min_queries_complex
        instruction = f"Generate {min_queries_complex}-{min_queries_complex + 5} queries for comprehensive analysis"

    return f"""
You are simulating Google's AI Mode query fan-out process.
Original query: "{q}"
Mode: "{mode}"

{instruction}

Include these query types:
1. Reformulations
2. Related Queries  
3. Implicit Queries
4. Comparative Queries
5. Entity Expansions
6. Personalized Queries

Return only valid JSON:
{{
  "generation_details": {{
    "target_query_count": {target},
    "reasoning_for_count": "Brief reasoning for number of queries"
  }},
  "expanded_queries": [
    {{
      "query": "Example query",
      "type": "reformulation",
      "user_intent": "Intent description",
      "reasoning": "Why this query was generated"
    }}
  ]
}}
"""

def generate_fanout(query, mode):
    """Generate query fan-out using Gemini"""
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        return data.get("expanded_queries", []), data.get("generation_details", {})
    except Exception as e:
        st.error(f"Error generating fan-out: {e}")
        return None, None

# Main App with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Fan-Out", "📄 Content Analysis", "🔬 Research & Fact-Check", "📊 Quick Fact Checker"])

# TAB 1: Query Fan-Out (Original functionality)
with tab1:
    st.header("🔍 Query Fan-Out Simulator")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area("Enter your query", "Why to Invest in Bangalore", height=100)
        mode = st.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

    with col2:
        st.subheader("📊 API Usage")
        st.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
        st.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

    if st.button("🚀 Generate Query Fan-Out", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Generating query fan-out..."):
                results, details = generate_fanout(user_query, mode)

            if results:
                st.session_state.fanout_results = results
                st.success(f"✅ Generated {len(results)} queries!")

                if details:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Queries", details.get('target_query_count', 'N/A'))
                    with col2:
                        st.metric("Generated", len(results))
                    with col3:
                        st.metric("Success Rate", "100%")
                    
                    st.info(f"**Reasoning:** {details.get('reasoning_for_count', 'Not provided')}")

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", data=csv, file_name="fanout_queries.csv", mime="text/csv")
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", data=json_data, file_name="fanout_queries.json", mime="application/json")

# TAB 2: Content Analysis (New feature)
with tab2:
    st.header("📄 Content Analysis & Enhancement")
    
    content_input_method = st.radio(
        "Choose content input method:",
        ["🔗 URL Analysis", "📝 Paste Content Directly"]
    )
    
    if content_input_method == "🔗 URL Analysis":
        st.subheader("🔗 URL Content Analysis")
        
        url_input = st.text_input(
            "Enter the URL to analyze:",
            placeholder="https://example.com/article",
            help="Enter the full URL of the article you want to analyze"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_url_btn = st.button("🔍 Analyze URL", type="primary")
        with col2:
            use_fallback = st.checkbox("Use manual input fallback", help="Check this if URL scraping fails")
        
        if analyze_url_btn and url_input:
            with st.spinner("Scraping and analyzing content..."):
                # Try to scrape URL
                scraped_data, error = scrape_url_content(url_input, use_fallback)
                
                if scraped_data:
                    content = scraped_data['content']
                    metadata = scraped_data['metadata']
                    
                    # Display metadata
                    st.success("✅ Content successfully scraped!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Word Count", metadata['word_count'])
                    with col2:
                        st.metric("Characters", metadata['char_count'])
                    with col3:
                        st.metric("Title", "Found" if metadata['title'] != 'No title found' else "Missing")
                    with col4:
                        st.metric("Scraped", "✅")
                    
                    # Store content for analysis
                    st.session_state.scraped_content = content
                    st.session_state.content_metadata = metadata
                    
                else:
                    st.error(f"❌ Failed to scrape content: {error}")
                    st.info("💡 Please use the 'Paste Content Directly' option below")
                    
                    # Automatic fallback
                    content_input_method = "📝 Paste Content Directly"
        
        # Show fallback text area if scraping failed
        if 'scraped_content' not in st.session_state and content_input_method == "🔗 URL Analysis":
            st.markdown("---")
            st.subheader("📝 Manual Content Input (Fallback)")
            st.info("If URL scraping doesn't work, paste the content manually:")
            
            fallback_content = st.text_area(
                "Paste article content here:",
                height=300,
                placeholder="Paste the article content here..."
            )
            
            if st.button("📊 Analyze Pasted Content") and fallback_content:
                st.session_state.scraped_content = fallback_content
                st.session_state.content_metadata = {
                    'word_count': len(fallback_content.split()),
                    'char_count': len(fallback_content),
                    'title': 'Manual Input',
                    'url': url_input or 'Manual Input',
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
    
    else:  # Direct content input
        st.subheader("📝 Direct Content Analysis")
        
        direct_content = st.text_area(
            "Paste your article content here:",
            height=300,
            placeholder="Paste the article content you want to analyze and enhance..."
        )
        
        if st.button("📊 Analyze Content", type="primary") and direct_content:
            st.session_state.scraped_content = direct_content
            st.session_state.content_metadata = {
                'word_count': len(direct_content.split()),
                'char_count': len(direct_content),
                'title': 'Direct Input',
                'url': 'Direct Input',
                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # Content Analysis Section
    if 'scraped_content' in st.session_state:
        st.markdown("---")
        st.subheader("🧠 Content Analysis Results")
        
        content = st.session_state.scraped_content
        metadata = st.session_state.content_metadata
        
        # Show content preview
        with st.expander("👀 Content Preview"):
            st.markdown(f"**Title:** {metadata['title']}")
            st.markdown(f"**Source:** {metadata['url']}")
            st.text_area("Content Preview", content[:1000] + "..." if len(content) > 1000 else content, height=200, disabled=True)
        
        if st.button("🔍 Perform Deep Analysis", type="secondary"):
            with st.spinner("Analyzing content structure and generating enhancement suggestions..."):
                # Analyze content structure
                analysis = analyze_content_structure(content)
                
                if analysis:
                    st.session_state.content_analysis = analysis
                    
                    # Generate enhancement suggestions
                    suggestions = generate_enhancement_suggestions(content, analysis)
                    if suggestions:
                        st.session_state.enhancement_suggestions = suggestions
        
        # Display Analysis Results
        if 'content_analysis' in st.session_state and st.session_state.content_analysis:
            analysis = st.session_state.content_analysis
            
            st.markdown("### 📋 Content Structure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Content Metrics**")
                metrics_df = pd.DataFrame([
                    {"Metric": "Content Type", "Value": analysis.get('content_type', 'N/A')},
                    {"Metric": "Tone", "Value": analysis.get('tone', 'N/A')},
                    {"Metric": "Target Audience", "Value": analysis.get('target_audience', 'N/A')},
                    {"Metric": "Readability", "Value": analysis.get('readability_level', 'N/A')},
                    {"Metric": "Writing Style", "Value": analysis.get('writing_style', 'N/A')}
                ])
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("**📝 Content Topics**")
                topics = analysis.get('main_topics', [])
                for i, topic in enumerate(topics[:5], 1):
                    st.markdown(f"{i}. {topic}")
                
                st.markdown("**🎯 Key Sections**")
                sections = analysis.get('key_sections', [])
                for section in sections[:3]:
                    st.markdown(f"• {section}")
            
            # Content Strengths
            strengths = analysis.get('strengths', [])
            if strengths:
                st.markdown("**✅ Content Strengths**")
                for strength in strengths:
                    st.markdown(f"""
                    <div class="enhancement-suggestion">
                        <strong>Strength:</strong> {strength}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Content Gaps
            gaps = analysis.get('content_gaps', [])
            if gaps:
                st.markdown("**⚠️ Content Gaps Identified**")
                for gap in gaps:
                    st.markdown(f"""
                    <div class="missing-topic">
                        <strong>Missing:</strong> {gap}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display Enhancement Suggestions
        if 'enhancement_suggestions' in st.session_state and st.session_state.enhancement_suggestions:
            suggestions = st.session_state.enhancement_suggestions
            
            st.markdown("---")
            st.markdown("### 🚀 Enhancement Suggestions")
            
            # Missing Data Points
            missing_data = suggestions.get('missing_data_points', [])
            if missing_data:
                st.markdown("#### 📊 Missing Data Points")
                for data_point in missing_data:
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(data_point.get('priority', 'medium'), "🟡")
                    st.markdown(f"""
                    <div class="content-analysis">
                        <strong>{priority_color} {data_point.get('category', 'Data Point')}</strong><br>
                        <strong>Suggestion:</strong> {data_point.get('suggestion', '')}<br>
                        <strong>Example:</strong> {data_point.get('example', '')}<br>
                        <strong>Why:</strong> {data_point.get('reasoning', '')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Topic Expansions
            topic_expansions = suggestions.get('topic_expansions', [])
            if topic_expansions:
                st.markdown("#### 📈 Topic Expansion Opportunities")
                for expansion in topic_expansions:
                    current_coverage = expansion.get('current_coverage', 'unknown')
                    coverage_emoji = {"brief": "📝", "moderate": "📄", "detailed": "📚"}.get(current_coverage, "📝")
                    
                    st.markdown(f"""
                    <div class="content-analysis">
                        <strong>{coverage_emoji} Topic: {expansion.get('topic', '')}</strong><br>
                        <strong>Current Coverage:</strong> {current_coverage}<br>
                        <strong>Suggested Additions:</strong>
                    """, unsafe_allow_html=True)
                    
                    additions = expansion.get('suggested_additions', [])
                    for addition in additions:
                        st.markdown(f"• {addition}")
                    
                    subtopics = expansion.get('potential_subtopics', [])
                    if subtopics:
                        st.markdown("**Potential Subtopics:**")
                        for subtopic in subtopics:
                            st.markdown(f"• {subtopic}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Content Improvements
            content_improvements = suggestions.get('content_improvements', [])
            if content_improvements:
                st.markdown("#### ✨ Content Improvements")
                for improvement in content_improvements:
                    area_emoji = {
                        "Structure": "🏗️", 
                        "Data": "📊", 
                        "Examples": "💡", 
                        "Sources": "📚"
                    }.get(improvement.get('area', ''), "🔧")
                    
                    st.markdown(f"""
                    <div class="enhancement-suggestion">
                        <strong>{area_emoji} {improvement.get('area', 'Improvement')}</strong><br>
                        <strong>Improvement:</strong> {improvement.get('improvement', '')}<br>
                        <strong>How to implement:</strong> {improvement.get('implementation', '')}<br>
                        <strong>Expected impact:</strong> {improvement.get('impact', '')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # SEO Enhancements
            seo_enhancements = suggestions.get('seo_enhancements', [])
            if seo_enhancements:
                st.markdown("#### 🔍 SEO Enhancement Opportunities")
                for seo in seo_enhancements:
                    seo_emoji = {
                        "Keywords": "🔑", 
                        "Headers": "📋", 
                        "Meta": "🏷️"
                    }.get(seo.get('type', ''), "🔍")
                    
                    st.markdown(f"""
                    <div class="content-analysis">
                        <strong>{seo_emoji} {seo.get('type', 'SEO')}</strong><br>
                        <strong>Suggestion:</strong> {seo.get('suggestion', '')}<br>
                        <strong>Implementation:</strong> {seo.get('implementation', '')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Fact Check Needed
            fact_checks = suggestions.get('fact_check_needed', [])
            if fact_checks:
                st.markdown("#### 🔍 Claims Requiring Fact-Checking")
                for fact in fact_checks:
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(fact.get('priority', 'medium'), "🟡")
                    st.markdown(f"""
                    <div class="missing-topic">
                        <strong>{priority_color} Claim to Verify:</strong> {fact.get('claim', '')}<br>
                        <strong>Reason:</strong> {fact.get('reason', '')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export Enhancement Report
            st.markdown("#### 📤 Export Enhancement Report")
            
            # Create comprehensive report
            enhancement_report = {
                'content_metadata': st.session_state.content_metadata,
                'analysis': st.session_state.content_analysis,
                'suggestions': st.session_state.enhancement_suggestions,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON Export
                report_json = json.dumps(enhancement_report, indent=2, default=str)
                st.download_button(
                    "📋 Download Analysis JSON",
                    data=report_json,
                    file_name=f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV Export for data points
                export_data = []
                for suggestion_type, suggestions_list in enhancement_report['suggestions'].items():
                    if isinstance(suggestions_list, list):
                        for item in suggestions_list:
                            if isinstance(item, dict):
                                export_data.append({
                                    'Type': suggestion_type,
                                    'Category': item.get('category', item.get('area', item.get('type', 'N/A'))),
                                    'Suggestion': item.get('suggestion', item.get('improvement', item.get('topic', 'N/A'))),
                                    'Priority': item.get('priority', 'N/A'),
                                    'Implementation': item.get('implementation', item.get('reasoning', 'N/A'))
                                })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv_data = export_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📊 Download Suggestions CSV",
                        data=csv_data,
                        file_name=f"enhancement_suggestions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Content Brief
                brief = f"""# Content Enhancement Brief

**Original Source:** {enhancement_report['content_metadata']['url']}
**Analysis Date:** {enhancement_report['generated_at']}
**Word Count:** {enhancement_report['content_metadata']['word_count']}

## Content Analysis Summary

**Content Type:** {enhancement_report['analysis'].get('content_type', 'N/A')}
**Target Audience:** {enhancement_report['analysis'].get('target_audience', 'N/A')}
**Tone:** {enhancement_report['analysis'].get('tone', 'N/A')}

## Key Enhancement Opportunities

### High-Priority Data Points to Add
"""
                
                # Add missing data points
                for data_point in enhancement_report['suggestions'].get('missing_data_points', []):
                    if data_point.get('priority') == 'high':
                        brief += f"- {data_point.get('suggestion', '')}: {data_point.get('example', '')}\n"
                
                brief += "\n### Topics to Expand\n"
                for expansion in enhancement_report['suggestions'].get('topic_expansions', []):
                    brief += f"- **{expansion.get('topic', '')}**: Currently {expansion.get('current_coverage', 'unknown')} coverage\n"
                    for addition in expansion.get('suggested_additions', []):
                        brief += f"  - {addition}\n"
                
                brief += "\n### SEO Improvements\n"
                for seo in enhancement_report['suggestions'].get('seo_enhancements', []):
                    brief += f"- {seo.get('type', '')}: {seo.get('suggestion', '')}\n"
                
                brief += "\n### Facts to Verify\n"
                for fact in enhancement_report['suggestions'].get('fact_check_needed', []):
                    brief += f"- {fact.get('claim', '')}\n"
                
                st.download_button(
                    "📝 Enhancement Brief",
                    data=brief,
                    file_name=f"enhancement_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )

# TAB 3: Research & Fact-Check (Enhanced with content-aware research)
with tab3:
    st.header("🔬 Research & Fact-Check Queries")
    
    # Show different options based on available data
    research_source = st.radio(
        "Research based on:",
        ["🔍 Fan-out queries", "📄 Content analysis suggestions", "✍️ Custom queries"]
    )
    
    if research_source == "🔍 Fan-out queries":
        if st.session_state.fanout_results:
            query_options = [f"{i+1}. {q['query']}" for i, q in enumerate(st.session_state.fanout_results)]
            selected_queries = st.multiselect(
                "Select queries to research (max 10 for cost control):",
                options=query_options,
                default=query_options[:5] if len(query_options) > 5 else query_options
            )
        else:
            st.info("No fan-out queries available. Please generate queries in the Query Fan-Out tab first.")
            selected_queries = []
    
    elif research_source == "📄 Content analysis suggestions":
        if 'enhancement_suggestions' in st.session_state and st.session_state.enhancement_suggestions:
            suggestions = st.session_state.enhancement_suggestions
            
            # Create research queries from enhancement suggestions
            research_queries = []
            
            # From missing data points
            for data_point in suggestions.get('missing_data_points', []):
                query = f"Latest data on {data_point.get('category', '')}: {data_point.get('suggestion', '')}"
                research_queries.append(query)
            
            # From topic expansions
            for expansion in suggestions.get('topic_expansions', []):
                for addition in expansion.get('suggested_additions', []):
                    query = f"Research {expansion.get('topic', '')}: {addition}"
                    research_queries.append(query)
            
            # From fact checks
            for fact in suggestions.get('fact_check_needed', []):
                query = f"Verify: {fact.get('claim', '')}"
                research_queries.append(query)
            
            if research_queries:
                selected_queries = st.multiselect(
                    "Select enhancement-based research queries (max 10):",
                    options=research_queries,
                    default=research_queries[:5] if len(research_queries) > 5 else research_queries
                )
            else:
                st.info("No content analysis suggestions available. Please analyze content in the Content Analysis tab first.")
                selected_queries = []
        else:
            st.info("No content analysis available. Please analyze content in the Content Analysis tab first.")
            selected_queries = []
    
    else:  # Custom queries
        st.subheader("✍️ Custom Research Queries")
        custom_queries_text = st.text_area(
            "Enter custom research queries (one per line):",
            placeholder="Latest Bangalore real estate prices 2024\nBangalore infrastructure development projects\nBangalore job market growth statistics",
            height=150
        )
        
        if custom_queries_text:
            custom_queries = [q.strip() for q in custom_queries_text.split('\n') if q.strip()]
            selected_queries = st.multiselect(
                "Select custom queries to research:",
                options=custom_queries,
                default=custom_queries[:5] if len(custom_queries) > 5 else custom_queries
            )
        else:
            selected_queries = []
    
    # Research configuration
    if selected_queries:
        research_focus = st.selectbox("Research Focus", [
            "Market Data & Statistics",
            "Current Facts & Trends", 
            "Investment Information",
            "Comparative Analysis",
            "Growth & Financial Data",
            "Content Enhancement Data"
        ])
        
        if st.button("🔍 Start Research & Fact-Check", type="secondary"):
            if len(selected_queries) > 10:
                st.warning("Limited to 10 queries to control API costs. Please select fewer queries.")
            else:
                research_results = []
                progress_bar = st.progress(0)
                
                for i, query in enumerate(selected_queries):
                    # Extract clean query text
                    if query.startswith(tuple("123456789")):
                        query_text = query.split('. ', 1)[1] if '. ' in query else query
                    else:
                        query_text = query
                    
                    progress_bar.progress((i + 1) / len(selected_queries))
                    
                    with st.spinner(f"Researching: {query_text[:50]}..."):
                        research_prompt = f"""
                        Research this query focusing on {research_focus}: {query_text}
                        
                        Provide:
                        1. Key facts with specific numbers and statistics
                        2. Current market data and trends
                        3. Recent developments and changes
                        4. Credible sources and references
                        
                        Focus on actionable information with numerical data.
                        """
                        
                        response = call_perplexity_api(research_prompt)
                        
                        if 'choices' in response:
                            content = response['choices'][0]['message']['content']
                            data_points = extract_data_points(content)
                            
                            research_results.append({
                                'query': query_text,
                                'research_content': content,
                                'data_points': data_points,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                            })
                        else:
                            research_results.append({
                                'query': query_text,
                                'research_content': f"Error: {response.get('error', 'Unknown error')}",
                                'data_points': [],
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                            })
                    
                    time.sleep(1)  # Rate limiting
                
                progress_bar.progress(1.0)
                st.session_state.research_data = research_results
                
                # Display Research Results
                st.success(f"✅ Research completed for {len(research_results)} queries!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Queries Researched", len(research_results))
                with col2:
                    total_data_points = sum(len(r['data_points']) for r in research_results)
                    st.metric("Data Points Found", total_data_points)
                with col3:
                    successful = sum(1 for r in research_results if 'Error:' not in r['research_content'])
                    st.metric("Success Rate", f"{(successful/len(research_results)*100):.0f}%")
                with col4:
                    st.metric("Research Focus", research_focus.split('&')[0])
                
                # Research Results Display
                st.subheader("📊 Research Results")
                
                for i, result in enumerate(research_results):
                    with st.expander(f"📋 {result['query'][:80]}..."):
                        st.markdown("**Research Findings:**")
                        st.markdown(result['research_content'])
                        
                        if result['data_points']:
                            st.markdown("**📊 Key Data Points:**")
                            
                            df_data = []
                            for dp in result['data_points']:
                                df_data.append({
                                    'Value': dp['value'],
                                    'Type': dp['type'],
                                    'Description': dp['description']
                                })
                            
                            if df_data:
                                data_df = pd.DataFrame(df_data)
                                st.dataframe(
                                    data_df,
                                    use_container_width=True,
                                    column_config={
                                        "Value": st.column_config.TextColumn("Value", width="small"),
                                        "Type": st.column_config.TextColumn("Type", width="small"),
                                        "Description": st.column_config.TextColumn("Context/Description", width="large")
                                    },
                                    hide_index=True
                                )
                        else:
                            st.info("No specific data points extracted from this research.")
                        
                        st.caption(f"Researched on: {result['timestamp']}")
                
                # Export Research Data
                st.subheader("📤 Export Research Data")
                
                export_data = []
                for result in research_results:
                    for dp in result.get('data_points', []):
                        export_data.append({
                            'Research_Query': result['query'],
                            'Data_Value': dp['value'],
                            'Data_Type': dp['type'],
                            'Context_Description': dp['description'],
                            'Research_Focus': research_focus,
                            'Timestamp': result['timestamp']
                        })
                    
                    if not result.get('data_points'):
                        export_data.append({
                            'Research_Query': result['query'],
                            'Data_Value': 'No data extracted',
                            'Data_Type': 'N/A',
                            'Context_Description': result['research_content'][:200] + '...',
                            'Research_Focus': research_focus,
                            'Timestamp': result['timestamp']
                        })
                
                research_df = pd.DataFrame(export_data)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    research_csv = research_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📊 Download Research CSV",
                        data=research_csv,
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    research_json = json.dumps(research_results, indent=2, default=str)
                    st.download_button(
                        "📋 Download Research JSON", 
                        data=research_json,
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    brief = f"""# Research Brief

**Research Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Focus Area:** {research_focus}
**Queries Researched:** {len(research_results)}

## Research Summary

"""
                    for result in research_results:
                        brief += f"\n### {result['query']}\n"
                        brief += f"{result['research_content'][:300]}...\n"
                        
                        if result.get('data_points'):
                            brief += f"\n**Key Data Found:**\n"
                            for dp in result['data_points'][:5]:
                                brief += f"- **{dp['value']}** ({dp['type']}): {dp['description'][:100]}...\n"
                        brief += "\n---\n"
                    
                    st.download_button(
                        "📝 Research Brief",
                        data=brief,
                        file_name=f"research_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )

# TAB 4: Quick Fact Checker (Enhanced)
with tab4:
    st.header("🔍 Quick Fact Checker")

    fact_query = st.text_input("Enter a statement or topic to fact-check:", placeholder="e.g., Bangalore property prices increased by 15% in 2024")

    if st.button("🔍 Verify Facts") and fact_query:
        with st.spinner("Fact-checking via Perplexity..."):
            response = call_perplexity_answer_api(fact_query)

            if 'answer' in response:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Fact-Based Answer")
                    st.markdown(response['answer'])
                    
                    data_points = extract_data_points(response['answer'])
                    
                    if data_points:
                        st.subheader("📊 Extracted Data Points")
                        df_data = []
                        for dp in data_points:
                            df_data.append({
                                'Value': dp['value'],
                                'Type': dp['type'],
                                'Description': dp['description']
                            })
                        
                        fact_df = pd.DataFrame(df_data)
                        st.dataframe(
                            fact_df,
                            use_container_width=True,
                            column_config={
                                "Value": st.column_config.TextColumn("Value", width="small"),
                                "Type": st.column_config.TextColumn("Type", width="small"),
                                "Description": st.column_config.TextColumn("Context", width="large")
                            },
                            hide_index=True
                        )
                
                with col2:
                    st.subheader("📊 Summary")
                    st.metric("Data Points Found", len(data_points))
                    if data_points:
                        for dp in data_points[:3]:
                            st.markdown(f"• **{dp['value']}** ({dp['type']})")
                            
                if 'sources' in response and response['sources']:
                    st.subheader("🔗 Sources & References")
                    for source in response['sources']:
                        title = source.get("title", "Source")
                        url = source.get("url", "")
                        st.markdown(f"- [{title}]({url})")
            else:
                st.error(response.get("error", "Unknown error."))

# Footer with enhanced metrics and controls
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Gemini Calls", st.session_state.api_usage['gemini_calls'])
    st.metric("Total Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

with col2:
    estimated_cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
    st.metric("Estimated Cost", f"${estimated_cost:.3f}")

with col3:
    # Session summary
    features_used = []
    if st.session_state.fanout_results:
        features_used.append("Query Fan-out")
    if 'content_analysis' in st.session_state:
        features_used.append("Content Analysis")
    if st.session_state.research_data:
        features_used.append("Research")
    
    st.metric("Features Used", len(features_used))
    if features_used:
        st.caption("Active: " + ", ".join(features_used))

with col4:
    if st.button("🗑️ Clear All Data"):
        # Clear all session state
        for key in ['fanout_results', 'research_data', 'content_analysis', 'enhancement_suggestions', 'scraped_content', 'content_metadata']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

# Additional Features Section
st.markdown("---")
st.markdown("### 🎯 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View API Usage Details"):
        st.info(f"""
        **API Usage Summary:**
        - Gemini Calls: {st.session_state.api_usage['gemini_calls']}
        - Perplexity Calls: {st.session_state.api_usage['perplexity_calls']}
        - Estimated Cost: ${estimated_cost:.3f}
        
        **Rate Limits:**
        - Perplexity: ~100 requests/hour
        - Gemini: ~1000 requests/minute
        """)

with col2:
    if st.button("💡 Usage Tips"):
        st.info("""
        **Pro Tips:**
        1. Use URL analysis for competitor content research
        2. Generate fan-out queries for comprehensive topic coverage
        3. Cross-reference research data with fact-checker
        4. Export data for offline analysis
        5. Use content analysis to identify content gaps
        """)

with col3:
    if st.button("🔧 Troubleshooting"):
        st.info("""
        **Common Issues:**
        1. **URL Scraping Fails:** Use manual content input
        2. **API Errors:** Check API keys and rate limits
        3. **No Data Points:** Try different research focus
        4. **Analysis Fails:** Content might be too short
        5. **Rate Limited:** Wait and retry
        """)

st.markdown("---")
st.markdown("**Qforia Pro Enhanced v3.0** - Advanced Query Research & Content Analysis Tool")
st.markdown("*Features: Query Fan-out, URL Content Analysis, Enhancement Suggestions, Research Automation, Fact-Checking*")
