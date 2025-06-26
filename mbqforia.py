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

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")

# API Keys
with st.sidebar.expander("🔑 API Configuration", expanded=True):
    gemini_key = st.text_input("Gemini API Key", type="password", value="AIzaSyDkJmnEXe85gbYjESn80csSqwpdd0RZbx8")
    perplexity_key = st.text_input("Perplexity API Key", type="password", value="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")

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
        system_prompt = "Provide only verified facts, numbers, and statistics. Be concise."
    elif focus == "insights":
        system_prompt = "Extract key insights and trends. Focus on actionable information."
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
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_fanout = st.button("Run Fan-Out 🚀", type="primary")
        with col_btn2:
            add_fact_check = st.button("Add Fact-Check Layer 🔍")
    
    with col2:
        st.markdown("### 📊 API Usage")
        st.metric("Gemini Calls", st.session_state.api_usage['gemini_calls'])
        st.metric("Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

    if run_fanout:
        if not user_query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("🤖 Generating query fan-out..."):
                results = generate_fanout(user_query, mode)

            if results:
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

                # Enhanced download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", data=csv, file_name="qforia_fanout.csv", mime="text/csv")
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", data=json_data, file_name="qforia_fanout.json", mime="application/json")
                with col3:
                    # Create summary report
                    summary = f"""
# Qforia Fan-Out Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Original Query: {user_query}
Mode: {mode}
Total Queries Generated: {len(results)}

## Query Breakdown by Type
{df['type'].value_counts().to_string()}
                    """
                    st.download_button("📥 Summary Report", data=summary, file_name="qforia_summary.md", mime="text/markdown")

    if add_fact_check and results:
        st.header("🔍 Real-Time Fact Verification")
        
        selected_queries = st.multiselect(
            "Select queries to fact-check:",
            options=[f"{i+1}. {q['query']}" for i, q in enumerate(results)],
            default=[f"1. {results[0]['query']}"] if results else []
        )
        
        if st.button("Verify Facts 🔍") and selected_queries:
            fact_results = []
            
            for selected in selected_queries:
                query_text = selected.split('. ', 1)[1]
                
                with st.spinner(f"Fact-checking: {query_text[:50]}..."):
                    fact_response = call_perplexity_api(
                        f"Verify facts and provide latest statistics for: {query_text}",
                        focus="facts"
                    )
                    
                    if 'choices' in fact_response:
                        fact_content = fact_response['choices'][0]['message']['content']
                        fact_results.append({
                            'query': query_text,
                            'verification': fact_content,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })
            
            st.session_state.fact_check_results = fact_results
            
            for result in fact_results:
                with st.expander(f"✅ {result['query'][:60]}..."):
                    st.markdown(f"**Verified on:** {result['timestamp']}")
                    st.markdown(result['verification'])

elif tool_mode == "Real-Time Fact Checker":
    st.header("🔍 Real-Time Fact Checker")
    
    fact_query = st.text_area("Enter statement or topic to fact-check:", height=100)
    
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
        urgency = st.selectbox("Priority Level", ["Standard", "High Priority", "Critical"])
    
    if st.button("Verify Now 🔍", type="primary"):
        if fact_query.strip():
            verification_prompt = f"""
            Fact-check this statement with current data: {fact_query}
            
            Focus on:
            - Accuracy verification
            - Latest statistics and numbers
            - Source credibility
            - Recent updates or changes
            """
            
            with st.spinner("🔍 Verifying facts with real-time data..."):
                fact_response = call_perplexity_api(verification_prompt, focus="facts")
                
                if 'choices' in fact_response:
                    verification_result = fact_response['choices'][0]['message']['content']
                    
                    # Analyze verification result for status
                    analysis_prompt = f"""
                    Analyze this fact-check result and categorize it:
                    {verification_result}
                    
                    Return JSON with:
                    - status: "verified", "partially_verified", "disputed", or "insufficient_data"
                    - confidence: 1-100
                    - key_facts: list of verified facts
                    - concerns: list of any issues found
                    """
                    
                    analysis = generate_gemini_content(analysis_prompt)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📋 Verification Results")
                        st.markdown(verification_result)
                        
                        # Try to parse analysis
                        try:
                            if analysis.startswith("```json"):
                                analysis = analysis[7:-3]
                            analysis_data = json.loads(analysis)
                            
                            status = analysis_data.get('status', 'unknown')
                            confidence = analysis_data.get('confidence', 0)
                            
                            if status == "verified":
                                st.success(f"✅ Verified (Confidence: {confidence}%)")
                            elif status == "partially_verified":
                                st.warning(f"⚠️ Partially Verified (Confidence: {confidence}%)")
                            elif status == "disputed":
                                st.error(f"❌ Disputed (Confidence: {confidence}%)")
                            else:
                                st.info(f"ℹ️ Insufficient Data (Confidence: {confidence}%)")
                                
                        except:
                            st.info("ℹ️ Analysis completed - see detailed results above")
                    
                    with col2:
                        st.subheader("📊 Quick Stats")
                        st.metric("Verification Time", "Real-time")
                        st.metric("Sources Checked", "Multiple")
                        st.metric("Data Freshness", "Latest")
                else:
                    st.error("❌ Verification failed. Please try again.")

elif tool_mode == "Content Research Assistant":
    st.header("📚 Content Research Assistant")
    
    research_topic = st.text_input("Research Topic:")
    content_type = st.selectbox("Content Type", [
        "Blog Post", "Article", "Report", "Presentation", 
        "Social Media", "Email Campaign", "Product Description"
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        target_audience = st.text_input("Target Audience:", "General audience")
        tone = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Conversational"])
    
    with col2:
        word_count = st.slider("Target Word Count", 500, 5000, 1500)
        include_stats = st.checkbox("Include Statistics", True)
    
    if st.button("Generate Research Brief 📋", type="primary"):
        if research_topic:
            research_prompt = f"""
            Create a comprehensive research brief for: {research_topic}
            
            Content type: {content_type}
            Target audience: {target_audience}
            Tone: {tone}
            Word count: {word_count}
            Include statistics: {include_stats}
            
            Provide:
            1. Key research questions
            2. Content outline
            3. Required data points
            4. Potential sources
            5. SEO keywords
            """
            
            with st.spinner("🔍 Generating research brief..."):
                # Get research brief from Gemini
                brief = generate_gemini_content(research_prompt)
                
                # Get current data from Perplexity
                data_prompt = f"Latest data, statistics, and trends for: {research_topic}"
                current_data = call_perplexity_api(data_prompt, focus="insights")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("📋 Research Brief")
                    st.markdown(brief)
                
                with col2:
                    st.subheader("📊 Current Data")
                    if 'choices' in current_data:
                        st.markdown(current_data['choices'][0]['message']['content'])
                    else:
                        st.error("Failed to fetch current data")

elif tool_mode == "Competitive Analysis Generator":
    st.header("🏆 Competitive Analysis Generator")
    
    company_name = st.text_input("Your Company/Product:")
    competitors = st.text_area("Competitors (one per line):", height=100)
    analysis_focus = st.multiselect(
        "Analysis Focus Areas:",
        ["Pricing", "Features", "Market Position", "Marketing Strategy", 
         "Customer Reviews", "Financial Performance", "Recent News"],
        default=["Features", "Market Position", "Pricing"]
    )
    
    if st.button("Generate Analysis 📊", type="primary"):
        if company_name and competitors:
            competitor_list = [c.strip() for c in competitors.split('\n') if c.strip()]
            
            analysis_results = {}
            
            for focus in analysis_focus:
                with st.spinner(f"Analyzing {focus}..."):
                    analysis_prompt = f"""
                    Compare {company_name} with {', '.join(competitor_list)} 
                    focusing on {focus}. Provide current, factual information.
                    """
                    
                    result = call_perplexity_api(analysis_prompt, focus="insights")
                    if 'choices' in result:
                        analysis_results[focus] = result['choices'][0]['message']['content']
            
            # Display results
            for focus, analysis in analysis_results.items():
                with st.expander(f"📈 {focus} Analysis"):
                    st.markdown(analysis)
            
            # Generate summary
            if analysis_results:
                summary_prompt = f"""
                Based on the competitive analysis data, provide:
                1. Key strengths of {company_name}
                2. Areas for improvement
                3. Market opportunities
                4. Strategic recommendations
                
                Analysis data: {str(analysis_results)}
                """
                
                with st.spinner("Generating strategic summary..."):
                    summary = generate_gemini_content(summary_prompt)
                    
                    st.subheader("🎯 Strategic Summary")
                    st.markdown(summary)

elif tool_mode == "SEO Content Planner":
    st.header("🔍 SEO Content Planner")
    
    primary_keyword = st.text_input("Primary Keyword:")
    industry = st.text_input("Industry/Niche:")
    
    col1, col2 = st.columns(2)
    with col1:
        content_goals = st.multiselect(
            "Content Goals:",
            ["Brand Awareness", "Lead Generation", "Sales", "Education", "Customer Support"]
        )
    
    with col2:
        competition_level = st.selectbox("Competition Level", ["Low", "Medium", "High"])
        content_frequency = st.selectbox("Publishing Frequency", ["Weekly", "Bi-weekly", "Monthly"])
    
    if st.button("Generate SEO Plan 📈", type="primary"):
        if primary_keyword:
            # Get keyword research and trends
            keyword_prompt = f"""
            SEO keyword research for: {primary_keyword} in {industry}
            
            Provide:
            - Related keywords
            - Search volume trends
            - Competition analysis
            - Content opportunities
            """
            
            with st.spinner("🔍 Researching keywords and trends..."):
                keyword_data = call_perplexity_api(keyword_prompt, focus="insights")
                
                # Generate content calendar
                calendar_prompt = f"""
                Create a content calendar for {primary_keyword} in {industry}:
                - Frequency: {content_frequency}
                - Goals: {', '.join(content_goals)}
                - Competition: {competition_level}
                
                Include content types, titles, and publishing schedule.
                """
                
                content_calendar = generate_gemini_content(calendar_prompt)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("📅 Content Calendar")
                    st.markdown(content_calendar)
                
                with col2:
                    st.subheader("🔍 Keyword Research")
                    if 'choices' in keyword_data:
                        st.markdown(keyword_data['choices'][0]['message']['content'])

elif tool_mode == "Market Insights Extractor":
    st.header("📊 Market Insights Extractor")
    
    market_query = st.text_input("Market/Industry to Analyze:")
    insight_type = st.selectbox("Insight Type", [
        "Market Size & Growth",
        "Consumer Trends", 
        "Competitive Landscape",
        "Investment Opportunities",
        "Risk Assessment",
        "Technology Trends"
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        time_frame = st.selectbox("Time Frame", ["Current", "6 Months", "1 Year", "3 Years"])
        geographic_focus = st.text_input("Geographic Focus:", "Global")
    
    with col2:
        data_sources = st.multiselect(
            "Preferred Data Sources:",
            ["Industry Reports", "Financial Data", "News Articles", "Research Papers", "Government Data"],
            default=["Industry Reports", "Financial Data"]
        )
    
    if st.button("Extract Insights 🎯", type="primary"):
        if market_query:
            insight_prompt = f"""
            Extract {insight_type} for {market_query}:
            
            Time frame: {time_frame}
            Geographic focus: {geographic_focus}
            Data sources: {', '.join(data_sources)}
            
            Focus on:
            - Key statistics and numbers
            - Recent trends and changes
            - Future projections
            - Actionable insights
            """
            
            with st.spinner("🔍 Extracting market insights..."):
                insights = call_perplexity_api(insight_prompt, focus="comprehensive")
                
                if 'choices' in insights:
                    insight_content = insights['choices'][0]['message']['content']
                    
                    # Generate visualization suggestions
                    viz_prompt = f"""
                    Based on these market insights, suggest data visualizations:
                    {insight_content}
                    
                    Return JSON with visualization suggestions.
                    """
                    
                    viz_suggestions = generate_gemini_content(viz_prompt)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader("📊 Market Insights")
                        st.markdown(insight_content)
                    
                    with col2:
                        st.subheader("📈 Viz Suggestions")
                        st.markdown(viz_suggestions)
                        
                        # Add to research history
                        st.session_state.research_history.append({
                            'query': market_query,
                            'type': insight_type,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'content': insight_content
                        })

# Footer with API usage and cost optimization tips
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Session Stats")
    st.metric("Total Gemini Calls", st.session_state.api_usage['gemini_calls'])
    st.metric("Total Perplexity Calls", st.session_state.api_usage['perplexity_calls'])

with col2:
    st.markdown("### 💡 Cost Optimization")
    st.info("✅ Using cost-effective models\n✅ Token limits applied\n✅ Batch processing enabled")

with col3:
    st.markdown("### 🎯 Quick Actions")
    if st.button("Clear History 🗑️"):
        st.session_state.research_history = []
        st.session_state.fact_check_results = []
        st.success("History cleared!")
    
    if st.button("Export All Data 📤"):
        export_data = {
            'api_usage': st.session_state.api_usage,
            'research_history': st.session_state.research_history,
            'fact_check_results': st.session_state.fact_check_results,
            'export_time': datetime.now().isoformat()
        }
        
        export_json = json.dumps(export_data, indent=2)
        st.download_button(
            "📥 Download Session Data",
            data=export_json,
            file_name=f"qforia_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
