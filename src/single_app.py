import streamlit as st
import joblib
import json
import torch
import numpy as np
import pandas as pd
import gc
from transformers import AutoModel, AutoTokenizer

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

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
    st.title("🧠 Sentiment Analysis")
    
    # Initialize session state for text input if not exists
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Create two columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # TEXT INPUT
        st.subheader("Enter Text for Analysis")
        text_input = st.text_area(
            "Paste your text here:",
            height=150,
            placeholder="Enter text to analyze sentiment...",
            key="text_input",
            value=st.session_state.text_input,
            label_visibility="collapsed"
        )
        
        # ANALYZE BUTTON
        analyze_button = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            key="analyze_btn"
        )
        
        # EXAMPLE TEXTS
        with st.expander("Try Example Texts"):
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                st.button(
                    "Positive Example",
                    on_click=set_positive_example,
                    use_container_width=True,
                    key="btn_positive"
                )
            
            with col_ex2:
                st.button(
                    "Negative Example",
                    on_click=set_negative_example,
                    use_container_width=True,
                    key="btn_negative"
                )
    
    with col_right:
        # RESULTS DISPLAY
        st.subheader("Results")
        
        if analyze_button and text_input.strip():
            try:
                # Show loading spinner
                with st.spinner("Analyzing sentiment..."):
                    # Load models (cached)
                    classifier, config = load_classifier()
                    tokenizer, model, device, _ = get_tokenizer_and_model()
                    
                    # Check if models loaded successfully
                    if tokenizer is None or model is None:
                        st.error("❌ Failed to load models.")
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
                if sentiment == "positive":
                    st.success(f"✅ POSITIVE SENTIMENT")
                else:
                    st.error(f"❌ NEGATIVE SENTIMENT")
                
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show probabilities
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("Negative", f"{probabilities[0]:.2%}")
                with col_prob2:
                    st.metric("Positive", f"{probabilities[1]:.2%}")
                    
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            # PLACEHOLDER FOR RESULTS
            st.info("Enter text and click 'Analyze Sentiment' to see results")

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
    main()