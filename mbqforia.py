import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import requests
from datetime import datetime
import time
from typing import List, Dict, Any

# App config
st.set_page_config(page_title="Qforia Pro", layout="wide")

# Simple CSS
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Qforia Pro: Query Fan-Out & Research Tool</h1>
    <p>AI-Powered Query Expansion with Real-Time Fact Verification</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = []
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}

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

# Add API testing section
if st.sidebar.button("🧪 Test API Connection"):
    if perplexity_key and perplexity_key.startswith('pplx-'):
        with st.sidebar.spinner("Testing Perplexity API..."):
            test_response = call_perplexity_api("What is the capital of France?")
            if 'choices' in test_response:
                st.sidebar.success("✅ Perplexity API working")
                st.sidebar.info(f"Model used: {test_response.get('model', 'sonar-pro')}")
            else:
                st.sidebar.error(f"❌ Perplexity API error: {test_response.get('error', 'Unknown')}")
    else:
        st.sidebar.error("❌ Invalid Perplexity API key")

# Model information
st.sidebar.subheader("📋 Available Models")
st.sidebar.markdown("""
**Current Models:**
- `sonar-pro` (Recommended)
- `llama-3.1-sonar-large-128k-online`
- `llama-3.1-sonar-small-128k-online`

**Note:** Model availability may change.
Check [Perplexity Docs](https://docs.perplexity.ai/guides/model-cards) for updates.
""")

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

# API Functions
def call_perplexity_api(query):
    """Call Perplexity API for research"""
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        # Use the correct current model name
        data = {
            "model": "sonar-pro",  # Updated to current valid model
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
            # Try with alternative model if sonar-pro doesn't work
            alternative_models = [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online"
            ]
            
            for model in alternative_models:
                simple_data = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": query}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2
                }
                
                simple_response = requests.post("https://api.perplexity.ai/chat/completions", 
                                              headers=headers, json=simple_data, timeout=30)
                
                if simple_response.status_code == 200:
                    st.session_state.api_usage['perplexity_calls'] += 1
                    return simple_response.json()
            
            # If all models fail, return error
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
            "model": "sonar-pro",  # Updated to current valid model
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
            
            # Extract answer from response
            if 'choices' in result and result['choices']:
                answer = result['choices'][0]['message']['content']
                return {"answer": answer, "sources": []}  # Simplified response
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
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        # Look for patterns with context
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
                
                # Clean up the sentence for context
                context = sentence.replace('\n', ' ').strip()
                if len(context) > 150:
                    # Find the position of the match and extract surrounding context
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
    
    # Remove duplicates while preserving order
    seen = set()
    unique_data_points = []
    for dp in data_points:
        identifier = (dp['value'], dp['type'])
        if identifier not in seen:
            seen.add(identifier)
            unique_data_points.append(dp)
    
    return unique_data_points

# Query Fan-Out Function (Original)
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

# Main App
st.header("🔍 Query Fan-Out Simulator")

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_area("Enter your query", "Why to Invest in Bangalore", height=100)
    mode = st.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

with col2:
    st.subheader("📊 API Usage")
    st.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
    st.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

# Step 1: Generate Fan-Out
if st.button("🚀 Generate Query Fan-Out", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating query fan-out..."):
            results, details = generate_fanout(user_query, mode)

        if results:
            st.session_state.fanout_results = results
            st.success(f"✅ Generated {len(results)} queries!")

            # Show generation details
            if details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Queries", details.get('target_query_count', 'N/A'))
                with col2:
                    st.metric("Generated", len(results))
                with col3:
                    st.metric("Success Rate", "100%")
                
                st.info(f"**Reasoning:** {details.get('reasoning_for_count', 'Not provided')}")

            # Display results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="fanout_queries.csv", mime="text/csv")
            with col2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", data=json_data, file_name="fanout_queries.json", mime="application/json")

# Step 2: Research & Fact-Check
if st.session_state.fanout_results:
    st.markdown("---")
    st.header("🔬 Research & Fact-Check Queries")
    
    # Select queries to research
    query_options = [f"{i+1}. {q['query']}" for i, q in enumerate(st.session_state.fanout_results)]
    selected_queries = st.multiselect(
        "Select queries to research (max 10 for cost control):",
        options=query_options,
        default=query_options[:5] if len(query_options) > 5 else query_options
    )
    
    research_focus = st.selectbox("Research Focus", [
        "Market Data & Statistics",
        "Current Facts & Trends", 
        "Investment Information",
        "Comparative Analysis",
        "Growth & Financial Data"
    ])
    
    if st.button("🔍 Start Research & Fact-Check", type="secondary") and selected_queries:
        if len(selected_queries) > 10:
            st.warning("Limited to 10 queries to control API costs. Please select fewer queries.")
        else:
            research_results = []
            progress_bar = st.progress(0)
            
            for i, selected in enumerate(selected_queries):
                # Extract query text
                query_text = selected.split('. ', 1)[1]
                
                progress_bar.progress((i + 1) / len(selected_queries))
                
                with st.spinner(f"Researching: {query_text[:50]}..."):
                    # Research prompt
                    research_prompt = f"""
                    Research this query focusing on {research_focus}: {query_text}
                    
                    Provide:
                    1. Key facts with specific numbers and statistics
                    2. Current market data and trends
                    3. Recent developments and changes
                    4. Credible sources and references
                    
                    Focus on actionable information with numerical data.
                    """
                    
                    # Call Perplexity API
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
            
            # Create tabs for each researched query
            if research_results:
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
                        # Research content
                        st.markdown("**Research Findings:**")
                        st.markdown(result['research_content'])
                        
                        # Data points in table format
                        if result['data_points']:
                            st.markdown("**📊 Key Data Points:**")
                            
                            # Create DataFrame for better display
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
                
                # Create comprehensive dataset
                export_data = []
                for result in research_results:
                    for dp in result.get('data_points', []):
                        export_data.append({
                            'Original_Query': user_query,
                            'Research_Query': result['query'],
                            'Data_Value': dp['value'],
                            'Data_Type': dp['type'],
                            'Context_Description': dp['description'],
                            'Research_Focus': research_focus,
                            'Timestamp': result['timestamp']
                        })
                    
                    # Also add a summary row for queries without data points
                    if not result.get('data_points'):
                        export_data.append({
                            'Original_Query': user_query,
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
                    # CSV Export
                    research_csv = research_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📊 Download Research CSV",
                        data=research_csv,
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # JSON Export
                    research_json = json.dumps(research_results, indent=2, default=str)
                    st.download_button(
                        "📋 Download Research JSON", 
                        data=research_json,
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    # Writer's Brief
                    brief = f"""# Content Writer's Research Brief

**Original Topic:** {user_query}
**Research Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Focus Area:** {research_focus}
**Queries Researched:** {len(research_results)}

## Key Data Points Summary

"""
                    for result in research_results:
                        brief += f"\n### {result['query']}\n"
                        brief += f"{result['research_content'][:300]}...\n"
                        
                        if result.get('data_points'):
                            brief += f"\n**Key Data Found:**\n"
                            for dp in result['data_points'][:5]:  # Top 5 data points
                                brief += f"- **{dp['value']}** ({dp['type']}): {dp['description'][:100]}...\n"
                        brief += "\n---\n"
                    
                    st.download_button(
                        "📝 Writer's Brief",
                        data=brief,
                        file_name=f"writers_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )

# Quick Fact Checker with Perplexity
st.markdown("---")
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
                
                # Extract and display data points
                data_points = extract_data_points(response['answer'])
                
                # Show data points in table if found
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
                    for dp in data_points[:3]:  # Show first 3
                        st.markdown(f"• **{dp['value']}** ({dp['type']})")
                        
            if 'sources' in response and response['sources']:
                st.subheader("🔗 Sources & References")
                for source in response['sources']:
                    title = source.get("title", "Source")
                    url = source.get("url", "")
                    st.markdown(f"- [{title}]({url})")
        else:
            st.error(response.get("error", "Unknown error."))

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Gemini Calls", st.session_state.api_usage['gemini_calls'])
    st.metric("Total Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

with col2:
    estimated_cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
    st.metric("Estimated Cost", f"${estimated_cost:.3f}")

with col3:
    if st.button("🗑️ Clear All Data"):
        st.session_state.fanout_results = None
        st.session_state.research_data = []
        st.success("Data cleared!")

st.markdown("---")
st.markdown("**Qforia Pro v2.0** - Simple, Fast, Effective Query Research Tool")
