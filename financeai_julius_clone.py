import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import openai
import io
import json
import time
import uuid
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="FinanceAI - Your AI Data Analyst",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= MODERN GREEN CSS STYLING (Julius AI INSPIRED) =============
st.markdown("""
<style>
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #f8fffe 0%, #f0f8f5 100%);
        padding: 0;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem 2rem 1rem 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }

    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px dashed #10b981;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #059669;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        transform: translateY(-2px);
    }

    .upload-icon {
        font-size: 3rem;
        color: #10b981;
        margin-bottom: 1rem;
    }

    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);
    }

    .ai-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        padding: 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 1rem 0;
        max-width: 85%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .ai-message pre {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        border-left: 4px solid #10b981;
    }

    /* Metrics Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #10b981;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #10b981;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #10b981;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .feature-icon {
        font-size: 2.5rem;
        color: #10b981;
        margin-bottom: 1rem;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }

    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #d1d5db;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE INITIALIZATION =============
def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8].upper()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'data_info' not in st.session_state:
        st.session_state.data_info = None

# ============= AI ANALYSIS FUNCTIONS =============
def get_data_context(df):
    """Generate comprehensive data context for AI analysis"""
    if df is None:
        return ""

    try:
        context = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Numeric Summary:
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}

Sample Data (first 3 rows):
{df.head(3).to_string()}
"""
        return context
    except Exception as e:
        return f"Error generating context: {str(e)}"

def analyze_with_openai(prompt, data_context, api_key):
    """Advanced financial data analysis with OpenAI"""
    try:
        if not api_key:
            return "❌ Please configure your OpenAI API key in the sidebar."

        openai.api_key = api_key

        system_prompt = """You are FinanceAI, an expert financial data analyst and visualization specialist. 

Your capabilities:
1. Financial data analysis (ratios, trends, forecasting, risk assessment)
2. Statistical analysis (correlations, regressions, hypothesis testing)
3. Data visualization recommendations
4. Python code generation for analysis
5. Financial insights and actionable recommendations

When analyzing financial data:
- Provide clear, actionable insights
- Suggest relevant visualizations
- Generate Python code when needed
- Explain financial concepts in business terms
- Focus on practical business applications

Respond in Indonesian for better user understanding, but keep technical terms in English when appropriate."""

        full_prompt = f"""
Data Context:
{data_context}

User Question: {prompt}

Please provide:
1. Analysis summary
2. Key financial insights  
3. Visualization recommendations
4. Python code (if analysis needed)
5. Business recommendations

Be comprehensive but concise. Focus on actionable financial insights.
"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except ImportError:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

            return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            return "❌ Invalid API key. Please check your OpenAI API key."
        elif "quota" in error_msg.lower():
            return "❌ API quota exceeded. Please check your OpenAI billing."
        else:
            return f"❌ Error: {error_msg}"

# ============= VISUALIZATION FUNCTIONS =============
def create_financial_visualizations(df, chart_type):
    """Create financial-focused visualizations"""
    try:
        colors = ['#10b981', '#059669', '#047857', '#065f46', '#064e3b']

        if chart_type == "revenue_trend":
            if 'Date' in df.columns or 'date' in df.columns:
                date_col = 'Date' if 'Date' in df.columns else 'date'
                revenue_cols = [col for col in df.columns if any(x in col.lower() for x in ['revenue', 'sales', 'income'])]

                if revenue_cols:
                    fig = px.line(df, x=date_col, y=revenue_cols[0], 
                                title="Revenue Trend Analysis",
                                color_discrete_sequence=colors)
                    fig.update_layout(template='plotly_white', title_font_size=18, title_x=0.5)
                    return fig

        elif chart_type == "profitability":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               title="Profitability Analysis",
                               color_discrete_sequence=colors,
                               trendline="ols")
                fig.update_layout(template='plotly_white', title_font_size=18, title_x=0.5)
                return fig

        elif chart_type == "distribution":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0],
                                 title=f"Distribution: {numeric_cols[0]}",
                                 color_discrete_sequence=colors)
                fig.update_layout(template='plotly_white', title_font_size=18, title_x=0.5)
                return fig

        elif chart_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, 
                              title="Correlation Analysis",
                              color_continuous_scale=['#dc2626', '#ffffff', '#10b981'])
                fig.update_layout(title_font_size=18, title_x=0.5)
                return fig

        return None

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# ============= MAIN APPLICATION =============
def main():
    initialize_session()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💚 FinanceAI</h1>
        <p>Your intelligent financial data analyst - powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 Configuration")

        api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key for advanced analysis"
        )

        if api_key:
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please add your OpenAI API key")

        st.markdown("---")
        st.markdown(f"🔒 Session: {st.session_state.session_id}")

        if st.button("🗑️ Clear Session"):
            st.session_state.messages = []
            st.session_state.data = None
            st.session_state.data_info = None
            st.rerun()

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-icon">📊</div>
            <h3>Upload Your Financial Data</h3>
            <p>Support: CSV, Excel files up to 200MB</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx'],
            help="Upload financial data for AI-powered analysis"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.data = df
                st.session_state.data_info = {
                    'filename': uploaded_file.name,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'upload_time': datetime.now().strftime('%H:%M:%S')
                }

                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Data metrics
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{df.shape[0]:,}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col1_2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{df.shape[1]}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col1_3:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{missing_pct:.1f}%</div>
                        <div class="metric-label">Missing</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Quick Analysis</h3>
            <p>Get instant insights from your financial data</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.data is not None:
            st.markdown("### 🚀 Quick Actions")

            quick_actions = [
                "Analyze revenue trends",
                "Calculate financial ratios", 
                "Find correlations",
                "Identify outliers",
                "Generate summary report"
            ]

            for action in quick_actions:
                if st.button(action, key=f"quick_{action}"):
                    if api_key:
                        with st.spinner("🤖 AI is analyzing..."):
                            data_context = get_data_context(st.session_state.data)
                            response = analyze_with_openai(action, data_context, api_key)

                            st.session_state.messages.append({
                                "role": "user",
                                "content": action,
                                "timestamp": datetime.now().strftime('%H:%M:%S')
                            })

                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "timestamp": datetime.now().strftime('%H:%M:%S')
                            })

                            st.rerun()
                    else:
                        st.warning("Please configure your OpenAI API key first!")

    # Chat Interface
    if st.session_state.data is not None:
        st.markdown("""
        <div class="chat-container">
            <h2>💬 Chat with Your Data</h2>
            <p>Ask questions about your financial data in plain language</p>
        </div>
        """, unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 1rem 1.5rem; border-radius: 18px 18px 5px 18px; margin: 1rem 0; max-width: 80%; margin-left: auto; box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);">
                    <strong>You ({message["timestamp"]})</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; color: #334155; padding: 1.5rem; border-radius: 18px 18px 18px 5px; margin: 1rem 0; max-width: 85%; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <strong>💚 FinanceAI ({message["timestamp"]})</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("Ask anything about your financial data...")

        if user_input and api_key:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime('%H:%M:%S')
            })

            with st.spinner("🤖 FinanceAI is analyzing your data..."):
                data_context = get_data_context(st.session_state.data)
                response = analyze_with_openai(user_input, data_context, api_key)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response, 
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })

            st.rerun()

    else:
        st.markdown("""
        <div class="chat-container" style="text-align: center; padding: 4rem 2rem;">
            <h2>🚀 Get Started in Seconds</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Upload your financial data and start getting AI-powered insights immediately</p>

            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div class="feature-card" style="max-width: 300px;">
                    <div class="feature-icon">📊</div>
                    <h4>Smart Analysis</h4>
                    <p>AI automatically detects patterns and generates insights</p>
                </div>

                <div class="feature-card" style="max-width: 300px;">
                    <div class="feature-icon">💬</div>
                    <h4>Natural Language</h4>
                    <p>Ask questions in plain English, get detailed answers</p>
                </div>

                <div class="feature-card" style="max-width: 300px;">
                    <div class="feature-icon">📈</div>
                    <h4>Instant Visualization</h4>
                    <p>Beautiful charts and graphs generated automatically</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Visualization Section
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("## 📈 Smart Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            chart_type = st.selectbox(
                "Choose Analysis Type:",
                ["revenue_trend", "profitability", "distribution", "correlation"],
                format_func=lambda x: {
                    "revenue_trend": "📈 Revenue Trend",
                    "profitability": "💰 Profitability Analysis", 
                    "distribution": "📊 Distribution Analysis",
                    "correlation": "🔗 Correlation Analysis"
                }[x]
            )

        with viz_col2:
            if st.button("🎨 Generate Visualization", use_container_width=True):
                with st.spinner("Creating visualization..."):
                    fig = create_financial_visualizations(st.session_state.data, chart_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to create visualization. Please check your data format.")

    # Footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 3rem 0 1rem 0;">
        <h3>💚 FinanceAI - Julius AI Clone for Finance</h3>
        <p>Created by Vito Devara | Phone: 081259795994</p>
        <p>Intelligent financial data analysis powered by OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
