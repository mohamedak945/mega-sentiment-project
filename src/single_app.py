# single_app.py - Streamlit app that calls the FastAPI service
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")  # Change this in production

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Health Check
    st.subheader("API Status")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.api_status = "✅ API is healthy"
            else:
                st.session_state.api_status = "❌ API is not responding"
        except Exception as e:
            st.session_state.api_status = f"❌ Connection failed: {str(e)}"
    
    if st.session_state.api_status:
        st.info(st.session_state.api_status)
    
    # Clear history
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    
    # About section
    st.subheader("ℹ️ About")
    st.info("""
    This dashboard analyzes sentiment using:
    - 🤖 Transformer models (XLMR)
    - 📊 CatBoost classifier
    - ⚡ FastAPI backend
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📈 History", "📊 Statistics"])

with tab1:
    # Text input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=150
        )
    
    with col2:
        st.markdown("### Examples")
        examples = [
            "I love this product! It's amazing!",
            "This is the worst experience I've ever had.",
            "The service was okay, nothing special.",
            "Absolutely terrible customer support!"
        ]
        
        for example in examples:
            if st.button(example[:40] + "..." if len(example) > 40 else example, 
                        key=f"example_{examples.index(example)}"):
                st.session_state.example_text = example
                st.rerun()
    
    # If example was clicked
    if 'example_text' in st.session_state:
        text_input = st.session_state.example_text
        del st.session_state.example_text
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Call FastAPI
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"text": text_input},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", result['sentiment'].upper())
                        
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        with col3:
                            st.metric("Label", result['label'])
                        
                        # Visual indicator
                        sentiment_color = "sentiment-positive" if result['sentiment'] == "positive" else "sentiment-negative"
                        st.markdown(f'<p class="{sentiment_color}">{result["sentiment"].upper()} sentiment detected</p>', 
                                  unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(result['confidence'])
                        
                        # Add to history
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": text_input[:100] + "..." if len(text_input) > 100 else text_input,
                            "sentiment": result['sentiment'],
                            "confidence": result['confidence'],
                            "label": result['label']
                        })
                        
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Make sure the FastAPI server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    if st.session_state.history:
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("Export as JSON"):
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("No analysis history yet. Start analyzing text in the 'Analyze' tab.")

with tab3:
    if st.session_state.history:
        # Statistics
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Analyses", len(df))
        
        with col2:
            positive_count = len(df[df['sentiment'] == 'positive'])
            st.metric("Positive", positive_count)
        
        with col3:
            negative_count = len(df[df['sentiment'] == 'negative'])
            st.metric("Negative", negative_count)
        
        # Charts
        st.subheader("📊 Sentiment Distribution")
        fig1 = px.pie(df, names='sentiment', title='Sentiment Distribution')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("📈 Confidence Over Time")
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt')
        fig2 = px.line(df, x='timestamp_dt', y='confidence', 
                      color='sentiment', title='Confidence Trend')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("📋 Detailed Statistics")
        st.write(df.describe())
    else:
        st.info("No data available for statistics yet.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, FastAPI, and Transformers</p>
    <p>API: {API_URL}</p>
</div>
""".format(API_URL=API_URL), unsafe_allow_html=True)