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
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value="AIzaSyDkJmnEXe85gbYjESn80csSqwpdd0RZbx8")
    perplexity_key = st.sidebar.text_input("Perplexity API Key", type="password", value="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")

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

# API Functions
def call_perplexity_api(query: str) -> Dict[str, Any]:
    """Call Perplexity API for fact-checking and research"""
    if not perplexity_key:
        return {"error": "Perplexity API key not provided"}
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "Provide factual information with specific numbers, statistics, and data points. Be concise but comprehensive."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        response.raise_for_status()
        st.session_state.api_usage['perplexity_calls'] += 1
        return response.json()
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

def extract_numbers(text: str) -> List[str]:
    """Extract numerical data from text"""
    patterns = [
        r'\d+(?:\.\d+)?%',  # Percentages
        r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?',  # Currency
        r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|crore|lakh)',  # Large numbers
        r'\d{4}(?:-\d{4})?',  # Years
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(matches)
    
    return list(set(numbers))  # Remove duplicates

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
                        numbers = extract_numbers(content)
                        
                        research_results.append({
                            'query': query_text,
                            'research_content': content,
                            'key_numbers': numbers,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })
                    else:
                        research_results.append({
                            'query': query_text,
                            'research_content': f"Error: {response.get('error', 'Unknown error')}",
                            'key_numbers': [],
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
                    total_numbers = sum(len(r['key_numbers']) for r in research_results)
                    st.metric("Data Points Found", total_numbers)
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
                        
                        # Key numbers
                        if result['key_numbers']:
                            st.markdown("**📊 Key Data Points:**")
                            for number in result['key_numbers']:
                                st.markdown(f"""
                                <div class="data-point">
                                    <span class="numerical-highlight">{number}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.caption(f"Researched on: {result['timestamp']}")
                
                # Export Research Data
                st.subheader("📤 Export Research Data")
                
                # Create comprehensive dataset
                export_data = []
                for result in research_results:
                    export_data.append({
                        'Original_Query': user_query,
                        'Research_Query': result['query'],
                        'Research_Content': result['research_content'],
                        'Key_Numbers': ', '.join(result['key_numbers']),
                        'Number_Count': len(result['key_numbers']),
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

## Key Findings Summary

"""
                    for result in research_results:
                        brief += f"\n### {result['query']}\n"
                        brief += f"{result['research_content'][:300]}...\n"
                        if result['key_numbers']:
                            brief += f"\n**Key Data:** {', '.join(result['key_numbers'][:5])}\n"
                        brief += "\n---\n"
                    
                    st.download_button(
                        "📝 Writer's Brief",
                        data=brief,
                        file_name=f"writers_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )

# Simple Fact Checker Tool
st.markdown("---")
st.header("🔍 Quick Fact Checker")

fact_query = st.text_input("Enter a statement to fact-check:", placeholder="e.g., Bangalore property prices increased by 15% in 2024")

if st.button("🔍 Verify Facts") and fact_query:
    with st.spinner("Fact-checking..."):
        fact_prompt = f"Fact-check this statement with current data and provide specific statistics: {fact_query}"
        response = call_perplexity_api(fact_prompt)
        
        if 'choices' in response:
            fact_result = response['choices'][0]['message']['content']
            numbers = extract_numbers(fact_result)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📋 Fact-Check Results")
                st.markdown(fact_result)
            
            with col2:
                st.subheader("📊 Data Found")
                st.metric("Numbers Extracted", len(numbers))
                if numbers:
                    for num in numbers[:5]:  # Show first 5
                        st.markdown(f"• **{num}**")
        else:
            st.error("Fact-check failed. Please try again.")

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
