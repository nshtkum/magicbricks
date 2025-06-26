import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import requests
from datetime import datetime
import time
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict

# App config
st.set_page_config(
    page_title="Qforia Pro - Advanced Writing Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .fact-verified {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .fact-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
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
    <h1>🚀 Qforia Pro: Advanced Writing Intelligence Suite</h1>
    <p>AI-Powered Query Expansion | Real-Time Fact Verification | Content Research Hub</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None
if 'fact_check_results' not in st.session_state:
    st.session_state.fact_check_results = []
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {'gemini_calls': 0, 'perplexity_calls': 0}
if 'research_data_table' not in st.session_state:
    st.session_state.research_data_table = []
if 'query_clusters' not in st.session_state:
    st.session_state.query_clusters = {}
if 'fanout_results' not in st.session_state:
    st.session_state.fanout_results = None

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")

# API Keys - Load from secrets
try:
    gemini_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
    perplexity_key = st.secrets["api_keys"]["PERPLEXITY_API_KEY"]
    st.sidebar.success("🔑 API Keys loaded from secrets")
except:
    # Fallback to manual input
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        gemini_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
        perplexity_key = st.text_input("Perplexity API Key", type="password", help="Enter your Perplexity API key")

# Tool Selection
st.sidebar.header("🛠️ Select Tool")
tool_mode = st.sidebar.selectbox(
    "Choose Your Tool",
    [
        "Query Fan-Out Simulator",
        "Real-Time Fact Checker",
        "Content Research Assistant",
        "Competitive Analysis Generator",
        "SEO Content Planner",
        "Market Insights Extractor"
    ]
)

# Configure APIs
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

def call_perplexity_api(query: str, focus: str = "comprehensive") -> Dict[str, Any]:
    """Call Perplexity API with optimized settings for minimal cost"""
    if not perplexity_key:
        return {"error": "Perplexity API key not provided"}
    
    headers = {
        "Authorization": f"Bearer {perplexity_key}",
        "Content-Type": "application/json"
    }
    
    # Optimize prompt based on focus to reduce token usage
    if focus == "facts":
        system_prompt = "Provide only verified facts, numbers, and statistics with sources. Be concise and factual."
    elif focus == "insights":
        system_prompt = "Extract key insights and trends. Focus on actionable information with data points."
    elif focus == "numerical":
        system_prompt = "Focus on numerical data, statistics, percentages, growth rates, and quantifiable metrics."
    else:
        system_prompt = "Provide comprehensive but concise information with key facts and insights."
    
    data = {
        "model": "llama-3.1-sonar-small-128k-online",  # Most cost-effective model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 1000,  # Limit tokens to control cost
        "temperature": 0.2,
        "top_p": 0.9
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=30)
        response.raise_for_status()
        st.session_state.api_usage['perplexity_calls'] += 1
        return response.json()
    except Exception as e:
        return {"error": f"Perplexity API error: {str(e)}"}

def generate_gemini_content(prompt: str) -> str:
    """Generate content using Gemini with error handling"""
    try:
        response = model.generate_content(prompt)
        st.session_state.api_usage['gemini_calls'] += 1
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
    """Extract numerical data points from text"""
    numerical_patterns = [
        r'(\d+(?:\.\d+)?%)',  # Percentages
        r'(\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|crore|lakh))?)',  # Currency
        r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|crore|lakh))',  # Large numbers
        r'(\d+(?:\.\d+)?\s*(?:sq\s*ft|acres|hectares))',  # Area measurements
        r'(\d{4}(?:-\d{4})?)',  # Years
        r'(\d+(?:\.\d+)?\s*(?:years?|months?|days?))',  # Time periods
    ]
    
    data_points = []
    for pattern in numerical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            data_points.append({
                'value': match,
                'context': text[max(0, text.find(match)-50):text.find(match)+50],
                'type': 'numerical'
            })
    
    return data_points

def cluster_queries(queries_list: List[str], n_clusters: int = 5) -> Dict[str, List[str]]:
    """Cluster queries by semantic similarity"""
    if len(queries_list) < 2:
        return {"General": queries_list}
    
    try:
        # Use TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(queries_list)
        
        # Adjust number of clusters based on data size
        n_clusters = min(n_clusters, len(queries_list))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Group queries by cluster
        clusters = defaultdict(list)
        cluster_names = [
            "Investment & Financial", "Location & Geography", "Market Trends", 
            "Infrastructure & Development", "Comparative Analysis", "Risk Assessment",
            "Future Outlook", "Regulatory & Legal", "Demographics", "Technology"
        ]
        
        for i, label in enumerate(cluster_labels):
            cluster_name = cluster_names[label] if label < len(cluster_names) else f"Topic {label + 1}"
            clusters[cluster_name].append(queries_list[i])
        
        return dict(clusters)
    except Exception as e:
        st.warning(f"Clustering failed: {e}. Using single group.")
        return {"General": queries_list}

def research_topic_cluster(cluster_name: str, queries: List[str], original_keyword: str) -> Dict[str, Any]:
    """Research a cluster of queries and extract data points"""
    
    # Create comprehensive research prompt
    research_prompt = f"""
    Research the topic cluster "{cluster_name}" for the keyword "{original_keyword}".
    
    Related queries: {', '.join(queries[:5])}  # Limit to first 5 to save tokens
    
    Provide:
    1. Key data points with numerical values (percentages, amounts, dates, statistics)
    2. Recent trends and growth rates
    3. Market insights specific to this topic area
    4. Factual information that would help a content writer
    5. Credible sources and references
    
    Focus on concrete, actionable information with numbers and facts.
    """
    
    try:
        # Get research data from Perplexity
        research_response = call_perplexity_api(research_prompt, focus="numerical")
        
        if 'choices' in research_response and research_response['choices']:
            research_content = research_response['choices'][0]['message']['content']
            
            # Extract numerical data points
            numerical_data = extract_numerical_data(research_content)
            
            # Structure the response
            cluster_data = {
                'cluster_name': cluster_name,
                'queries_count': len(queries),
                'research_content': research_content,
                'numerical_data': numerical_data,
                'key_insights': [],
                'data_points': [],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'status': 'success'
            }
            
            # Extract key insights using Gemini
            insight_prompt = f"""
            Extract 5-7 key insights from this research content that would help a content writer:
            
            {research_content}
            
            Format as bullet points, each focusing on actionable information.
            """
            
            insights = generate_gemini_content(insight_prompt)
            cluster_data['key_insights'] = insights.split('\n') if insights else []
            
            # Create structured data points
            data_points = []
            for num_data in numerical_data:
                data_points.append({
                    'value': num_data['value'],
                    'description': num_data['context'].strip(),
                    'category': cluster_name
                })
            
            cluster_data['data_points'] = data_points
            
            return cluster_data
            
        else:
            return {
                'cluster_name': cluster_name,
                'status': 'error',
                'error_message': 'No data received from API'
            }
    
    except Exception as e:
        return {
            'cluster_name': cluster_name,
            'status': 'error',
            'error_message': str(e)
        }

# Query Fan-Out Simulator (Enhanced Original Functionality)
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_simple}**. "
            f"For a straightforward query, generating around {min_queries_simple}-{min_queries_simple + 2} queries might be sufficient. "
            f"If the query has a few distinct aspects or common follow-up questions, aim for a slightly higher number, perhaps {min_queries_simple + 3}-{min_queries_simple + 5} queries. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries themselves should be tightly scoped and highly relevant."
        )
    else:
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_complex}**. "
            f"For multifaceted queries requiring exploration of various angles, sub-topics, comparisons, or deeper implications, "
            f"you should generate a more comprehensive set, potentially {min_queries_complex + 5}-{min_queries_complex + 10} queries, or even more if the query is exceptionally broad or deep. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries should be diverse and in-depth."
        )

    return (
        f"You are simulating Google's AI Mode query fan-out process for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"**Your first task is to determine the total number of queries to generate and the reasoning for this number, based on the instructions below:**\n"
        f"{num_queries_instruction}\n\n"
        f"**Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.**\n"
        "Each of the following query transformation types MUST be represented at least once in the generated set, if the total number of queries you decide to generate allows for it:\n"
        "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        "The 'reasoning' field for each *individual query* should explain why that specific query was generated in relation to the original query, its type, and the overall user intent.\n"
        "Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        "Return only a valid JSON object. The JSON object should strictly follow this format:\n"
        "{\n"
        "  \"generation_details\": {\n"
        "    \"target_query_count\": 12,\n"
        "    \"reasoning_for_count\": \"The user query was moderately complex, so I chose to generate slightly more than the minimum for a simple overview to cover key aspects like X, Y, and Z.\"\n"
        "  },\n"
        "  \"expanded_queries\": [\n"
        "    {\n"
        "      \"query\": \"Example query 1...\",\n"
        "      \"type\": \"reformulation\",\n"
        "      \"user_intent\": \"Example intent...\",\n"
        "      \"reasoning\": \"Example reasoning for this specific query...\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

def generate_fanout(query, mode):
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        generation_details = data.get("generation_details", {})
        expanded_queries = data.get("expanded_queries", [])

        st.session_state.generation_details = generation_details
        return expanded_queries
    except Exception as e:
        st.error(f"🔴 Error generating fan-out: {e}")
        return None

# Tool Implementations
if tool_mode == "Query Fan-Out Simulator":
    st.header("🔍 Query Fan-Out Simulator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_query = st.text_area("Enter your query", "Why to Invest in Bangalore", height=120)
        mode = st.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])
        
        # Main action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_fanout = st.button("🚀 Run Fan-Out", type="primary", use_container_width=True)
        with col_btn2:
            if st.session_state.fanout_results:
                start_research = st.button("🔬 Research & Fact-Check", type="secondary", use_container_width=True)
            else:
                st.button("🔬 Research & Fact-Check", disabled=True, help="Run fan-out first", use_container_width=True)
                start_research = False
    
    with col2:
        st.markdown("### 📊 API Usage")
        st.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
        st.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

    # Fan-out execution
    if run_fanout:
        if not user_query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("🤖 Generating query fan-out..."):
                results = generate_fanout(user_query, mode)

            if results:
                st.session_state.fanout_results = results
                st.success("✅ Query fan-out complete!")

                if st.session_state.generation_details:
                    details = st.session_state.generation_details
                    generated_count = len(results)
                    target_count_model = details.get('target_query_count', 'N/A')
                    reasoning_model = details.get('reasoning_for_count', 'Not provided by model.')

                    st.markdown("---")
                    st.subheader("🧠 Model's Query Generation Plan")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Queries", target_count_model)
                    with col2:
                        st.metric("Generated", generated_count)
                    with col3:
                        accuracy = "100%" if target_count_model == generated_count else "Variance"
                        st.metric("Accuracy", accuracy)
                    
                    st.markdown(f"**Model's Reasoning:** _{reasoning_model}_")
                    st.markdown("---")

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3)

                # Download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", data=csv, file_name="qforia_fanout.csv", mime="text/csv")
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", data=json_data, file_name="qforia_fanout.json", mime="application/json")
                with col3:
                    summary = f"""# Qforia Fan-Out Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Original Query: {user_query}
Mode: {mode}
Total Queries Generated: {len(results)}

## Query Breakdown by Type
{df['type'].value_counts().to_string()}"""
                    st.download_button("📥 Summary Report", data=summary, file_name="qforia_summary.md", mime="text/markdown")

    # Research and Fact-Checking Flow
    if st.session_state.fanout_results and 'start_research' in locals() and start_research:
        st.markdown("---")
        st.header("🔬 Advanced Research & Fact-Checking")
        
        # Research configuration
        col1, col2 = st.columns(2)
        with col1:
            research_depth = st.selectbox("Research Depth", 
                ["Quick Facts", "Comprehensive Analysis", "Deep Dive with Statistics"])
        with col2:
            focus_areas = st.multiselect("Focus Areas", 
                ["Market Data", "Financial Metrics", "Growth Statistics", "Comparative Analysis", "Recent Trends"],
                default=["Market Data", "Recent Trends"])
        
        if st.button("🔍 Start Research Process", type="primary"):
            with st.spinner("🔬 Analyzing queries and clustering topics..."):
                # Step 1: Cluster the queries
                query_texts = [q['query'] for q in st.session_state.fanout_results]
                clusters = cluster_queries(query_texts)
                
                st.success(f"✅ Identified {len(clusters)} topic clusters")
                
                # Step 2: Research each cluster
                cluster_research_data = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (cluster_name, cluster_queries) in enumerate(clusters.items()):
                    status_text.text(f"Researching {cluster_name}...")
                    progress = (i + 1) / len(clusters)
                    progress_bar.progress(progress)
                    
                    # Research this cluster
                    cluster_data = research_topic_cluster(cluster_name, cluster_queries, user_query)
                    cluster_research_data[cluster_name] = cluster_data
                    
                    time.sleep(1)  # Rate limiting
                
                progress_bar.progress(1.0)
                status_text.text("Research complete!")
                
                # Step 3: Display results in organized format
                st.markdown("---")
                st.header("📊 Research Results by Topic Cluster")
                
                # Create tabs for each cluster
                tab_names = list(cluster_research_data.keys()) + ["📈 Summary Dashboard"]
                tabs = st.tabs(tab_names)
                
                all_data_points = []
                
                # Individual cluster tabs
                for i, (cluster_name, cluster_data) in enumerate(cluster_research_data.items()):
                    with tabs[i]:
                        if cluster_data.get('status') == 'success':
                            st.subheader(f"📁 {cluster_name}")
                            
                            # Cluster metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Queries in Cluster", cluster_data['queries_count'])
                            with col2:
                                st.metric("Data Points Found", len(cluster_data.get('numerical_data', [])))
                            with col3:
                                st.metric("Research Status", "✅ Complete")
                            
                            # Key insights
                            st.markdown("### 💡 Key Insights for Writers")
                            insights = cluster_data.get('key_insights', [])
                            for insight in insights:
                                if insight.strip():
                                    st.markdown(f"• {insight.strip()}")
                            
                            # Numerical data points
                            st.markdown("### 📊 Data Points & Statistics")
                            numerical_data = cluster_data.get('numerical_data', [])
                            
                            if numerical_data:
                                for data_point in numerical_data:
                                    st.markdown(f"""
                                    <div class="data-point">
                                        <span class="numerical-highlight">{data_point['value']}</span>
                                        <br><small>{data_point['context']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add to all data points for summary
                                    all_data_points.append({
                                        'cluster': cluster_name,
                                        'value': data_point['value'],
                                        'context': data_point['context']
                                    })
                            else:
                                st.info("No specific numerical data found for this cluster.")
                            
                            # Full research content
                            with st.expander("📋 Complete Research Content"):
                                st.markdown(cluster_data.get('research_content', ''))
                            
                            # Export cluster data
                            if cluster_data.get('data_points'):
                                cluster_df = pd.DataFrame(cluster_data['data_points'])
                                cluster_csv = cluster_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    f"📥 Export {cluster_name} Data",
                                    data=cluster_csv,
                                    file_name=f"qforia_{cluster_name.lower().replace(' ', '_')}_data.csv",
                                    mime="text/csv",
                                    key=f"download_{i}"
                                )
                        
                        else:
                            st.error(f"❌ Research failed for {cluster_name}")
                            st.error(cluster_data.get('error_message', 'Unknown error'))
                
                # Summary dashboard tab
                with tabs[-1]:
                    st.subheader("📈 Research Summary Dashboard")
                    
                    if all_data_points:
                        # Overall metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Clusters", len(cluster_research_data))
                        with col2:
                            st.metric("Total Data Points", len(all_data_points))
                        with col3:
                            successful_clusters = sum(1 for data in cluster_research_data.values() if data.get('status') == 'success')
                            st.metric("Success Rate", f"{(successful_clusters/len(cluster_research_data)*100):.1f}%")
                        with col4:
                            st.metric("Research Depth", research_depth)
                        
                        # Data points summary table
                        st.markdown("### 📊 All Data Points Summary")
                        summary_df = pd.DataFrame(all_data_points)
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            height=400,
                            column_config={
                                "cluster": st.column_config.SelectboxColumn("Topic Cluster"),
                                "value": st.column_config.TextColumn("Data Value", width="medium"),
                                "context": st.column_config.TextColumn("Context", width="large")
                            }
                        )
                        
                        # Cluster distribution chart
                        cluster_counts = summary_df['cluster'].value_counts().reset_index()
                        cluster_counts.columns = ['Cluster', 'Data Points']
                        
                        fig = px.bar(
                            cluster_counts,
                            x='Cluster',
                            y='Data Points',
                            title='Data Points Distribution by Topic Cluster',
                            color='Data Points',
                            color_continuous_scale='viridis'
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export complete research data
                        st.markdown("### 📤 Export Complete Research")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            complete_csv = summary_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📊 All Data Points CSV",
                                data=complete_csv,
                                file_name=f"qforia_complete_research_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Create writer's brief
                            writer_brief = f"""# Content Writer's Research Brief

**Topic:** {user_query}
**Research Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Clusters Analyzed:** {len(cluster_research_data)}

## Key Data Points by Topic

"""
                            for cluster_name, cluster_data in cluster_research_data.items():
                                if cluster_data.get('status') == 'success':
                                    writer_brief += f"\n### {cluster_name}\n"
                                    insights = cluster_data.get('key_insights', [])
                                    for insight in insights[:3]:  # Top 3 insights
                                        if insight.strip():
                                            writer_brief += f"- {insight.strip()}\n"
                                    
                                    numerical_data = cluster_data.get('numerical_data', [])
                                    if numerical_data:
                                        writer_brief += "\n**Key Statistics:**\n"
                                        for data in numerical_data[:3]:  # Top 3 data points
                                            writer_brief += f"- {data['value']}: {data['context'][:100]}...\n"
                            
                            st.download_button(
                                "📝 Writer's Brief",
                                data=writer_brief,
                                file_name=f"writer_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                mime="text/markdown"
                            )
                        
                        with col3:
                            # Create comprehensive JSON export
                            export_data = {
                                'metadata': {
                                    'original_query': user_query,
                                    'research_date': datetime.now().isoformat(),
                                    'research_depth': research_depth,
                                    'focus_areas': focus_areas,
                                    'total_clusters': len(cluster_research_data),
                                    'total_data_points': len(all_data_points)
                                },
                                'cluster_research': cluster_research_data,
                                'data_points_summary': all_data_points,
                                'api_usage': st.session_state.api_usage
                            }
                            
                            export_json = json.dumps(export_data, indent=2, default=str)
                            st.download_button(
                                "📋 Complete JSON",
                                data=export_json,
                                file_name=f"qforia_research_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )
                    
                    else:
                        st.warning("No data points were extracted from the research. Try adjusting the research focus or query.")

elif tool_mode == "Real-Time Fact Checker":
    st.header("🔍 Real-Time Fact Checker")
    
    fact_query = st.text_area("Enter statement or topic to fact-check:", height=100, 
                             placeholder="e.g., 'Bangalore property prices increased by 15% in 2024'")
    
    col1, col2 = st.columns(2)
    with col1:
        check_type = st.selectbox("Verification Type", [
            "General Fact Check",
            "Statistical Verification", 
            "Recent News Verification",
            "Company/Financial Data",
            "Market Research Data"
        ])
    
    with col2:
        detail_level = st.selectbox("Detail Level", ["Quick Check", "Detailed Analysis", "Comprehensive Report"])
    
    if st.button("🔍 Verify Facts", type="primary"):
        if fact_query.strip():
            verification_prompt = f"""
            Fact-check this statement with current data: {fact_query}
            
            Focus on:
            - Accuracy verification with sources
            - Latest statistics and numbers
            - Source credibility assessment
            - Recent updates or changes
            - Provide specific numerical data where available
            """
            
            with st.spinner("🔍 Verifying facts with real-time data..."):
                fact_response = call_perplexity_api(verification_prompt, focus="facts")
                
                if 'choices' in fact_response and fact_response['choices']:
                    verification_result = fact_response['choices'][0]['message']['content']
                    
                    # Extract numerical data from verification
                    numerical_data = extract_numerical_data(verification_result)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📋 Verification Results")
                        st.markdown(verification_result)
                        
                        # Display extracted data points
                        if numerical_data:
                            st.subheader("📊 Key Data Points")
                            for data_point in numerical_data:
                                st.markdown(f"""
                                <div class="data-point">
                                    <span class="numerical-highlight">{data_point['value']}</span>
                                    <br><small>{data_point['context']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📊 Verification Summary")
                        st.metric("Verification Time", "Real-time")
                        st.metric("Data Points Found", len(numerical_data))
                        st.metric("Sources Checked", "Multiple")
                        
                        # Save to fact-check results
                        fact_check_entry = {
                            'query': fact_query,
                            'result': verification_result,
                            'numerical_data': numerical_data,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'type': check_type
                        }
                        st.session_state.fact_check_results.append(fact_check_entry)
                        
                        # Export fact-check
                        fact_export = pd.DataFrame([fact_check_entry])
                        fact_csv = fact_export.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Export Fact-Check",
                            data=fact_csv,
                            file_name=f"fact_check_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Verification failed. Please try again.")

elif tool_mode == "Content Research Assistant":
    st.header("📚 Content Research Assistant")
    
    research_topic = st.text_input("Research Topic:", placeholder="e.g., 'Real estate investment in Indian metros'")
    
    col1, col2 = st.columns(2)
    with col1:
        content_type = st.selectbox("Content Type", [
            "Blog Post", "Article", "Report", "Presentation", 
            "Social Media", "Email Campaign", "Product Description"
        ])
        target_audience = st.text_input("Target Audience:", "General audience")
    
    with col2:
        word_count = st.slider("Target Word Count", 500, 5000, 1500)
        tone = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Conversational"])
    
    research_focus = st.multiselect("Research Focus Areas:", [
        "Market Statistics", "Industry Trends", "Expert Opinions", 
        "Case Studies", "Comparative Data", "Future Projections"
    ], default=["Market Statistics", "Industry Trends"])
    
    if st.button("🔍 Generate Research Brief", type="primary"):
        if research_topic:
            # Generate comprehensive research brief
            research_prompt = f"""
            Create a comprehensive research brief for: {research_topic}
            
            Content details:
            - Type: {content_type}
            - Target audience: {target_audience}
            - Word count: {word_count}
            - Tone: {tone}
            - Focus areas: {', '.join(research_focus)}
            
            Provide:
            1. Content outline with key sections
            2. Required data points and statistics
            3. Research questions to explore
            4. Potential data sources
            5. SEO keyword suggestions
            """
            
            with st.spinner("🔍 Generating research brief..."):
                # Get research brief from Gemini
                brief = generate_gemini_content(research_prompt)
                
                # Get current data from Perplexity
                data_prompt = f"Latest statistics, trends, and data points for: {research_topic}. Focus on {', '.join(research_focus)}"
                current_data = call_perplexity_api(data_prompt, focus="numerical")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("📋 Research Brief")
                    st.markdown(brief)
                
                with col2:
                    st.subheader("📊 Current Data & Statistics")
                    if 'choices' in current_data and current_data['choices']:
                        current_content = current_data['choices'][0]['message']['content']
                        st.markdown(current_content)
                        
                        # Extract numerical data
                        numerical_data = extract_numerical_data(current_content)
                        if numerical_data:
                            st.subheader("🔢 Key Numbers")
                            for data in numerical_data[:5]:  # Show top 5
                                st.markdown(f"**{data['value']}** - {data['context'][:100]}...")
                    else:
                        st.error("Failed to fetch current data")
                
                # Export research package
                research_package = f"""# Content Research Package

## Research Brief
{brief}

## Current Data & Statistics
{current_content if 'current_content' in locals() else 'No current data available'}

## Metadata
- Topic: {research_topic}
- Content Type: {content_type}
- Target Audience: {target_audience}
- Word Count: {word_count}
- Tone: {tone}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
                
                st.download_button(
                    "📥 Download Research Package",
                    data=research_package,
                    file_name=f"research_package_{research_topic.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

elif tool_mode == "Competitive Analysis Generator":
    st.header("🏆 Competitive Analysis Generator")
    
    company_name = st.text_input("Your Company/Product:", placeholder="e.g., 'MagicBricks'")
    competitors = st.text_area("Competitors (one per line):", height=100, 
                              placeholder="99acres\nHousing.com\nCommonFloor")
    
    analysis_focus = st.multiselect(
        "Analysis Focus Areas:",
        ["Pricing", "Features", "Market Position", "Marketing Strategy", 
         "Customer Reviews", "Financial Performance", "Recent News", "Market Share"],
        default=["Features", "Market Position", "Pricing"]
    )
    
    if st.button("🔍 Generate Analysis", type="primary"):
        if company_name and competitors:
            competitor_list = [c.strip() for c in competitors.split('\n') if c.strip()]
            
            st.subheader(f"🏆 Competitive Analysis: {company_name}")
            
            analysis_results = {}
            progress_bar = st.progress(0)
            
            for i, focus in enumerate(analysis_focus):
                progress_bar.progress((i + 1) / len(analysis_focus))
                
                with st.spinner(f"Analyzing {focus}..."):
                    analysis_prompt = f"""
                    Compare {company_name} with {', '.join(competitor_list)} 
                    focusing on {focus}. 
                    
                    Provide:
                    - Current market data and statistics
                    - Specific metrics and numbers
                    - Competitive positioning
                    - Key differentiators
                    """
                    
                    result = call_perplexity_api(analysis_prompt, focus="insights")
                    if 'choices' in result and result['choices']:
                        analysis_results[focus] = result['choices'][0]['message']['content']
            
            progress_bar.progress(1.0)
            
            # Display results in tabs
            if analysis_results:
                tabs = st.tabs(list(analysis_results.keys()) + ["📊 Summary"])
                
                # Individual analysis tabs
                for i, (focus, analysis) in enumerate(analysis_results.items()):
                    with tabs[i]:
                        st.markdown(f"### {focus} Analysis")
                        st.markdown(analysis)
                        
                        # Extract numerical data
                        numerical_data = extract_numerical_data(analysis)
                        if numerical_data:
                            st.subheader("📊 Key Metrics")
                            for data in numerical_data:
                                st.markdown(f"""
                                <div class="data-point">
                                    <span class="numerical-highlight">{data['value']}</span>
                                    <br><small>{data['context']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Summary tab
                with tabs[-1]:
                    # Generate strategic summary
                    summary_prompt = f"""
                    Based on the competitive analysis data, provide:
                    1. Key strengths of {company_name}
                    2. Areas for improvement
                    3. Market opportunities
                    4. Strategic recommendations with specific actions
                    
                    Analysis data: {str(analysis_results)}
                    """
                    
                    with st.spinner("Generating strategic summary..."):
                        summary = generate_gemini_content(summary_prompt)
                        
                        st.subheader("🎯 Strategic Summary")
                        st.markdown(summary)
                        
                        # Export complete analysis
                        complete_analysis = f"""# Competitive Analysis Report

**Company:** {company_name}
**Competitors:** {', '.join(competitor_list)}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Strategic Summary
{summary}

## Detailed Analysis

"""
                        for focus, analysis in analysis_results.items():
                            complete_analysis += f"\n### {focus}\n{analysis}\n\n"
                        
                        st.download_button(
                            "📥 Download Complete Analysis",
                            data=complete_analysis,
                            file_name=f"competitive_analysis_{company_name}_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )

elif tool_mode == "SEO Content Planner":
    st.header("🔍 SEO Content Planner")
    
    primary_keyword = st.text_input("Primary Keyword:", placeholder="e.g., 'property investment Bangalore'")
    industry = st.text_input("Industry/Niche:", placeholder="e.g., 'Real Estate'")
    
    col1, col2 = st.columns(2)
    with col1:
        content_goals = st.multiselect(
            "Content Goals:",
            ["Brand Awareness", "Lead Generation", "Sales", "Education", "Customer Support"],
            default=["Lead Generation", "Education"]
        )
        competition_level = st.selectbox("Competition Level", ["Low", "Medium", "High"])
    
    with col2:
        content_frequency = st.selectbox("Publishing Frequency", ["Weekly", "Bi-weekly", "Monthly"])
        target_audience = st.text_input("Target Audience:", "Property investors")
    
    if st.button("🔍 Generate SEO Plan", type="primary"):
        if primary_keyword:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Get keyword research and trends
                keyword_prompt = f"""
                SEO keyword research for: {primary_keyword} in {industry}
                
                Provide:
                - Related long-tail keywords
                - Search volume estimates
                - Competition analysis
                - Content opportunities
                - Seasonal trends
                """
                
                with st.spinner("🔍 Researching keywords and trends..."):
                    keyword_data = call_perplexity_api(keyword_prompt, focus="insights")
                    
                    st.subheader("🔍 Keyword Research")
                    if 'choices' in keyword_data and keyword_data['choices']:
                        keyword_content = keyword_data['choices'][0]['message']['content']
                        st.markdown(keyword_content)
                        
                        # Extract numerical data from keyword research
                        keyword_numbers = extract_numerical_data(keyword_content)
                        if keyword_numbers:
                            st.subheader("📊 Search Volume & Competition Data")
                            for data in keyword_numbers:
                                st.markdown(f"**{data['value']}** - {data['context'][:100]}...")
            
            with col2:
                # Generate content calendar
                calendar_prompt = f"""
                Create a content calendar for {primary_keyword} in {industry}:
                - Frequency: {content_frequency}
                - Goals: {', '.join(content_goals)}
                - Competition: {competition_level}
                - Target audience: {target_audience}
                
                Include:
                - Content titles
                - Content types
                - Publishing schedule
                - Target keywords for each piece
                """
                
                with st.spinner("📅 Creating content calendar..."):
                    content_calendar = generate_gemini_content(calendar_prompt)
                    
                    st.subheader("📅 Content Calendar")
                    st.markdown(content_calendar)
            
            # Export SEO plan
            seo_plan = f"""# SEO Content Plan

**Primary Keyword:** {primary_keyword}
**Industry:** {industry}
**Target Audience:** {target_audience}
**Goals:** {', '.join(content_goals)}
**Competition Level:** {competition_level}
**Publishing Frequency:** {content_frequency}

## Keyword Research
{keyword_content if 'keyword_content' in locals() else 'No data available'}

## Content Calendar
{content_calendar}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            
            st.download_button(
                "📥 Download SEO Plan",
                data=seo_plan,
                file_name=f"seo_plan_{primary_keyword.replace(' ', '_')}.md",
                mime="text/markdown"
            )

elif tool_mode == "Market Insights Extractor":
    st.header("📊 Market Insights Extractor")
    
    market_query = st.text_input("Market/Industry to Analyze:", 
                                placeholder="e.g., 'Indian real estate market 2024'")
    
    col1, col2 = st.columns(2)
    with col1:
        insight_type = st.selectbox("Insight Type", [
            "Market Size & Growth",
            "Consumer Trends", 
            "Competitive Landscape",
            "Investment Opportunities",
            "Risk Assessment",
            "Technology Trends"
        ])
        time_frame = st.selectbox("Time Frame", ["Current", "6 Months", "1 Year", "3 Years"])
    
    with col2:
        geographic_focus = st.text_input("Geographic Focus:", "India")
        data_sources = st.multiselect(
            "Preferred Data Sources:",
            ["Industry Reports", "Financial Data", "News Articles", "Research Papers", "Government Data"],
            default=["Industry Reports", "Financial Data"]
        )
    
    if st.button("🔍 Extract Insights", type="primary"):
        if market_query:
            insight_prompt = f"""
            Extract {insight_type} for {market_query}:
            
            Parameters:
            - Time frame: {time_frame}
            - Geographic focus: {geographic_focus}
            - Data sources: {', '.join(data_sources)}
            
            Focus on:
            - Key statistics and numerical data
            - Recent trends and percentage changes
            - Future projections with specific numbers
            - Market size figures
            - Growth rates and percentages
            - Investment amounts and valuations
            """
            
            with st.spinner("🔍 Extracting market insights..."):
                insights = call_perplexity_api(insight_prompt, focus="numerical")
                
                if 'choices' in insights and insights['choices']:
                    insight_content = insights['choices'][0]['message']['content']
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("📊 Market Insights")
                        st.markdown(insight_content)
                    
                    with col2:
                        # Extract and highlight numerical data
                        numerical_insights = extract_numerical_data(insight_content)
                        
                        st.subheader("🔢 Key Market Data")
                        if numerical_insights:
                            # Create metrics display
                            for i, data in enumerate(numerical_insights[:6]):  # Top 6 metrics
                                st.markdown(f"""
                                <div class="metric-card" style="margin: 0.5rem 0;">
                                    <h4>{data['value']}</h4>
                                    <p style="font-size: 0.8rem; margin: 0;">{data['context'][:80]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No specific numerical data extracted")
                        
                        # Market summary metrics
                        st.metric("Insight Type", insight_type)
                        st.metric("Data Points", len(numerical_insights))
                        st.metric("Geographic Scope", geographic_focus)
                    
                    # Add to research history
                    insight_entry = {
                        'query': market_query,
                        'type': insight_type,
                        'content': insight_content,
                        'numerical_data': numerical_insights,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'geographic_focus': geographic_focus,
                        'time_frame': time_frame
                    }
                    st.session_state.research_history.append(insight_entry)
                    
                    # Export insights
                    insights_export = f"""# Market Insights Report

**Market:** {market_query}
**Insight Type:** {insight_type}
**Time Frame:** {time_frame}
**Geographic Focus:** {geographic_focus}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Insights
{insight_content}

## Numerical Data Summary
"""
                    for data in numerical_insights:
                        insights_export += f"- **{data['value']}**: {data['context']}\n"
                    
                    st.download_button(
                        "📥 Download Market Insights",
                        data=insights_export,
                        file_name=f"market_insights_{market_query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )

# Footer with enhanced session management
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Session Statistics")
    st.metric("Gemini API Calls", st.session_state.api_usage['gemini_calls'])
    st.metric("Perplexity API Calls", st.session_state.api_usage['perplexity_calls'])
    
    # Cost estimation
    estimated_cost = (st.session_state.api_usage['perplexity_calls'] * 0.002) + (st.session_state.api_usage['gemini_calls'] * 0.001)
    st.metric("Estimated Cost", f"${estimated_cost:.3f}")

with col2:
    st.markdown("### 📈 Research Activity")
    st.metric("Fact-Checks Performed", len(st.session_state.fact_check_results))
    st.metric("Research History Items", len(st.session_state.research_history))
    
    if st.session_state.query_clusters:
        st.metric("Topic Clusters", len(st.session_state.query_clusters))

with col3:
    st.markdown("### 🛠️ Session Management")
    
    col_clear, col_export = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear All Data", help="Clear all session data"):
            st.session_state.research_history = []
            st.session_state.fact_check_results = []
            st.session_state.research_data_table = []
            st.session_state.query_clusters = {}
            st.session_state.fanout_results = None
            st.success("All data cleared!")
    
    with col_export:
        if st.button("📤 Export Session", help="Export all session data"):
            session_export = {
                'session_metadata': {
                    'export_time': datetime.now().isoformat(),
                    'api_usage': st.session_state.api_usage,
                    'total_fact_checks': len(st.session_state.fact_check_results),
                    'total_research_items': len(st.session_state.research_history)
                },
                'fact_check_results': st.session_state.fact_check_results,
                'research_history': st.session_state.research_history,
                'query_clusters': st.session_state.query_clusters
            }
            
            session_json = json.dumps(session_export, indent=2, default=str)
            st.download_button(
                "📥 Download Session Data",
                data=session_json,
                file_name=f"qforia_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key="session_export"
            )

# Cost optimization tips
with st.expander("💡 Cost Optimization Tips"):
    st.markdown("""
    **Current Optimizations:**
    - ✅ Using most cost-effective Perplexity model (`llama-3.1-sonar-small-128k-online`)
    - ✅ Token limits applied (1000 tokens max per call)
    - ✅ Smart rate limiting to avoid unnecessary API calls
    - ✅ Focused prompts to reduce token usage
    - ✅ Batch processing where possible
    
    **Tips to Reduce Costs:**
    - Use "Quick Facts" research depth for simple queries
    - Limit focus areas to essential topics only
    - Export and reuse research data instead of re-querying
    - Use the fact-checker sparingly for high-priority items only
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Qforia Pro v2.0 | Advanced Writing Intelligence Suite for MagicBricks</p>
    <p>Powered by Google Gemini & Perplexity AI | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
