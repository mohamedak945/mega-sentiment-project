import streamlit as st
import joblib
import json
import torch
import numpy as np
import pandas as pd
import time
import gc
import sys
from transformers import AutoModel, AutoTokenizer

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER UI
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-left: 5px solid #4F46E5;
    }
    .positive {
        background-color: #D1FAE5;
        border-left-color: #10B981;
    }
    .negative {
        background-color: #FEE2E2;
        border-left-color: #EF4444;
    }
    .model-info {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4F46E5;
    }
    .example-btn {
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS WITH MEMORY OPTIMIZATION
# ============================================

@st.cache_resource(show_spinner=False, max_entries=1)
def load_config():
    """Load model configuration"""
    with open('artifacts/models/best_model_info.json', 'r') as f:
        config = json.load(f)
    return config

@st.cache_resource(show_spinner=False, max_entries=1)
def load_classifier():
    """Load classifier model with memory optimization"""
    config = load_config()
    model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
    
    # Clear memory before loading
    gc.collect()
    
    # Load classifier
    classifier = joblib.load(model_path)
    
    # Clear memory after loading
    gc.collect()
    
    return classifier, config

@st.cache_resource(show_spinner=False, max_entries=1)
def get_tokenizer_and_model():
    """Load tokenizer and model with memory optimization"""
    config = load_config()
    
    # Map embedding names to model names
    model_mapping = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'xlmr': 'xlm-roberta-base'
    }
    
    embedding_model_name = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
    
    # Clear memory before loading
    gc.collect()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name,
            use_fast=True  # Fast tokenizer uses less memory
        )
        
        # Force CPU usage to save memory
        device = torch.device("cpu")
        
        # Try loading with accelerate, fallback to standard loading
        try:
            # Option 1: With accelerate for low memory usage
            model = AutoModel.from_pretrained(
                embedding_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
        except:
            # Option 2: Standard loading without accelerate features
            st.warning("⚠️ Using standard model loading (accelerate not available)")
            model = AutoModel.from_pretrained(
                embedding_model_name,
                torch_dtype=torch.float32
            )
            model = model.to(device)
        
        # Disable gradients and set to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Clear memory after loading
        gc.collect()
        
        return tokenizer, model, device, config
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Return None values to prevent crash
        return None, None, None, config

# ============================================
# EMBEDDING FUNCTION
# ============================================

def get_embedding(text: str, tokenizer, model, device):
    """Convert text to embedding vector"""
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding.cpu().numpy()

# ============================================
# CALLBACK FUNCTIONS FOR EXAMPLE BUTTONS
# ============================================

def set_positive_example():
    """Callback for positive example button"""
    st.session_state.text_input = "This product is absolutely amazing! I've never been happier with a purchase. The quality is outstanding and it exceeded all my expectations."

def set_negative_example():
    """Callback for negative example button"""
    st.session_state.text_input = "I'm very disappointed with this service. The quality is poor and it doesn't work as advertised. Would not recommend to anyone."

# ============================================
# MAIN APP
# ============================================

def main():
    # HEADER
    st.markdown("<h1 class='main-header'>🧠 Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    # Initialize session state for text input if not exists
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # SIDEBAR - MODEL INFO
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        
        try:
            # Load config (lightweight)
            config = load_config()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Classifier", config['classifier'].upper())
            with col2:
                st.metric("Embedding", config['embedding_model'].upper())
            
            st.markdown("---")
            st.metric("Accuracy", f"{config['metrics']['accuracy']:.2%}")
            st.metric("F1 Score", f"{config['metrics']['f1_score']:.2%}")
            
            st.markdown("---")
            st.markdown("### 📈 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                'Value': [
                    config['metrics']['accuracy'],
                    config['metrics']['f1_score'],
                    config['metrics']['precision'],
                    config['metrics']['recall']
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error loading model info: {e}")
    
    # MAIN CONTENT
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # TEXT INPUT
        st.markdown("### ✍️ Enter Text for Analysis")
        text_input = st.text_area(
            "Paste your text here:",
            height=150,
            placeholder="Enter text to analyze sentiment...\n\nExample: 'I absolutely loved this product! It exceeded all my expectations.'",
            key="text_input",
            value=st.session_state.text_input
        )
        
        # ANALYZE BUTTON
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_button = st.button(
                "🔍 Analyze Sentiment",
                type="primary",
                use_container_width=True,
                key="analyze_btn"
            )
        
        # EXAMPLE TEXTS - FIXED VERSION
        with st.expander("📋 Try Example Texts"):
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                st.button(
                    "😊 Positive Example",
                    on_click=set_positive_example,
                    use_container_width=True,
                    key="btn_positive"
                )
            
            with col_ex2:
                st.button(
                    "😠 Negative Example",
                    on_click=set_negative_example,
                    use_container_width=True,
                    key="btn_negative"
                )
            
            # Additional examples using selectbox
            st.markdown("---")
            st.markdown("**Or choose from more examples:**")
            
            example_options = {
                "Select an example...": "",
                "📱 Tech Product": "The new smartphone has excellent battery life and a fantastic camera. However, the software could use some improvements.",
                "🏨 Hotel Stay": "The hotel staff was friendly and accommodating. The room was clean and comfortable, but the breakfast options were limited.",
                "🍽️ Restaurant": "The food was delicious and reasonably priced. Service was slow but the atmosphere made up for it.",
                "📚 Book Review": "The plot was engaging and characters were well-developed. Couldn't put it down until I finished!"
            }
            
            selected_example = st.selectbox(
                "Choose:",
                options=list(example_options.keys()),
                key="example_selector",
                label_visibility="collapsed"
            )
            
            if example_options[selected_example]:
                st.session_state.text_input = example_options[selected_example]
                st.rerun()
    
    with col_right:
        # RESULTS DISPLAY
        st.markdown("### 📊 Results")
        
        if analyze_button and text_input.strip():
            try:
                # Show loading spinner
                with st.spinner("Analyzing sentiment..."):
                    # Load models (cached)
                    classifier, config = load_classifier()
                    tokenizer, model, device, _ = get_tokenizer_and_model()
                    
                    # Check if models loaded successfully
                    if tokenizer is None or model is None:
                        st.error("❌ Failed to load models. Please check the console for errors.")
                        st.stop()
                    
                    # Generate embedding
                    embedding = get_embedding(text_input, tokenizer, model, device)
                    
                    # Make prediction
                    prediction = classifier.predict(embedding)[0]
                    probabilities = classifier.predict_proba(embedding)[0]
                    confidence = float(max(probabilities))
                    
                    sentiment = "positive" if prediction == 1 else "negative"
                    
                    # Clear memory after prediction
                    gc.collect()
                
                # DISPLAY RESULTS
                sentiment_color = "🟢" if sentiment == "positive" else "🔴"
                sentiment_class = "positive" if sentiment == "positive" else "negative"
                
                st.markdown(f"""
                <div class="result-box {sentiment_class}">
                    <h3>{sentiment_color} {sentiment.upper()} SENTIMENT</h3>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <p><strong>Label:</strong> {prediction} ({sentiment})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # CONFIDENCE BAR
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.2%}")
                
                # PROBABILITY DISTRIBUTION
                prob_data = pd.DataFrame({
                    'Sentiment': ['Negative', 'Positive'],
                    'Probability': [probabilities[0], probabilities[1]]
                })
                
                st.bar_chart(prob_data.set_index('Sentiment'))
                
                # DETAILS EXPANDER
                with st.expander("📈 View Detailed Probabilities"):
                    st.dataframe(prob_data, use_container_width=True, hide_index=True)
                    
                    # Show raw prediction values
                    st.markdown("**Raw Prediction Values:**")
                    st.json({
                        "prediction": int(prediction),
                        "probabilities": {
                            "negative": float(probabilities[0]),
                            "positive": float(probabilities[1])
                        },
                        "confidence": float(confidence)
                    })
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
                st.info("Try using a shorter text or check if models are loaded correctly.")
                
                # Debug info
                with st.expander("🔧 Debug Info"):
                    st.code(f"Error type: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            # PLACEHOLDER FOR RESULTS
            st.info("👈 Enter text and click 'Analyze Sentiment' to see results here")
            
            # Show example of what results will look like
            with st.expander("👀 What to expect"):
                st.markdown("""
                **After analysis, you'll see:**
                - ✅ Sentiment (Positive/Negative)
                - 📊 Confidence score
                - 📈 Probability distribution
                - 🔢 Raw prediction values
                
                **Example output:**
                - 🟢 **POSITIVE SENTIMENT** (85% confidence)
                - Label: 1 (positive)
                """)
    
    # FOOTER
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f2:
        st.caption("Built with ❤️ using Streamlit, Transformers, and Scikit-learn")
    
    # ADDITIONAL FEATURES
    with st.expander("📁 Batch Analysis (Upload CSV)"):
        uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'], key="csv_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.success(f"✅ File uploaded successfully! {len(df)} rows found.")
                    
                    if st.button("Analyze All Texts", key="batch_analyze"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        classifier, config = load_classifier()
                        tokenizer, model, device, _ = get_tokenizer_and_model()
                        
                        for i, text in enumerate(df['text']):
                            status_text.text(f"Processing row {i+1}/{len(df)}...")
                            progress_bar.progress((i + 1) / len(df))
                            
                            try:
                                embedding = get_embedding(str(text), tokenizer, model, device)
                                prediction = classifier.predict(embedding)[0]
                                probabilities = classifier.predict_proba(embedding)[0]
                                confidence = float(max(probabilities))
                                sentiment = "positive" if prediction == 1 else "negative"
                                
                                results.append({
                                    'text': text[:100] + "..." if len(str(text)) > 100 else text,
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'prediction': prediction
                                })
                            except Exception as e:
                                results.append({
                                    'text': text[:100] + "..." if len(str(text)) > 100 else text,
                                    'sentiment': 'error',
                                    'confidence': 0.0,
                                    'prediction': -1,
                                    'error': str(e)
                                })
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        if len(results_df) > 0:
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                positive_count = len(results_df[results_df['sentiment'] == 'positive'])
                                st.metric("Positive", positive_count)
                            with col_sum2:
                                negative_count = len(results_df[results_df['sentiment'] == 'negative'])
                                st.metric("Negative", negative_count)
                            with col_sum3:
                                avg_confidence = results_df[results_df['sentiment'].isin(['positive', 'negative'])]['confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_results.csv",
                            mime="text/csv",
                            key="download_results"
                        )
                else:
                    st.error("CSV must contain a 'text' column")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # MEMORY INFO (DEBUG)
    if st.sidebar.checkbox("Show Memory Info", False, key="debug_checkbox"):
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
            if st.sidebar.button("Clear Cache", key="clear_cache"):
                st.cache_resource.clear()
                gc.collect()
                st.sidebar.success("Cache cleared!")
                time.sleep(1)
                st.rerun()
        except:
            st.sidebar.warning("Could not get memory info")

# ============================================
# APP ENTRY POINT
# ============================================
if __name__ == "__main__":
    # Clear cache on startup
    if 'cache_cleared' not in st.session_state:
        try:
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.session_state.cache_cleared = True
        except:
            pass
    
    # Run main app
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please try refreshing the page or contact support.")
        
        # Show detailed error in expander
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())