import streamlit as st
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# LOAD BEST MODEL COMBO
# ============================================

@st.cache_resource
def load_best_model():
    """Load the best model combination from config"""
    # Load config
    with open('artifacts/models/best_model_info.json', 'r') as f:
        config = json.load(f)
    
    st.sidebar.info(f"**Best Model:** {config['classifier']} on {config['embedding_model']}")
    st.sidebar.info(f"**Accuracy:** {config['metrics']['accuracy']:.2%}")
    
    # Load classifier
    model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
    classifier = joblib.load(model_path)
    
    # Load embedding model
    model_mapping = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'xlmr': 'xlm-roberta-base'
    }
    
    embedding_model_name = model_mapping.get(config['embedding_model'], 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    
    # Use CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return classifier, tokenizer, model, device, config

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
# PREDICT FUNCTION
# ============================================

def predict(text: str):
    """Make sentiment prediction"""
    classifier, tokenizer, model, device, config = load_best_model()
    
    # Get embedding
    embedding = get_embedding(text, tokenizer, model, device)
    
    # Predict
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]
    
    # Determine sentiment (use probability-based decision)
    # If negative probability > positive probability, it's negative
    if probabilities[0] > probabilities[1]:
        sentiment = "negative"
        confidence = probabilities[0]
        label = 0
    else:
        sentiment = "positive"
        confidence = probabilities[1]
        label = 1
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "label": label,
        "probabilities": probabilities.tolist()
    }

# ============================================
# MAIN APP - SIMPLE AND CLEAN
# ============================================

def main():
    st.title("🧠 Sentiment Analysis")
    
    # Load model info
    classifier, _, _, _, config = load_best_model()
    
    # Text input
    st.subheader("Enter Text to Analyze")
    text = st.text_area(
        "",
        height=150,
        placeholder="Type your text here...",
        key="text_input"
    )
    
    # Example buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😊 Positive Example", use_container_width=True):
            st.session_state.text_input = "This product is absolutely amazing! I've never been happier with a purchase. The quality is outstanding!"
    with col2:
        if st.button("😠 Negative Example", use_container_width=True):
            st.session_state.text_input = "I'm very disappointed with this service. The quality is poor and it doesn't work as advertised."
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = predict(text)
            
            # Display results
            st.subheader("Results")
            
            if result["sentiment"] == "positive":
                st.success(f"✅ **POSITIVE** (Confidence: {result['confidence']:.2%})")
            else:
                st.error(f"❌ **NEGATIVE** (Confidence: {result['confidence']:.2%})")
            
            # Probabilities
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Negative", f"{result['probabilities'][0]:.2%}")
            with col_b:
                st.metric("Positive", f"{result['probabilities'][1]:.2%}")
            
            # Raw data
            with st.expander("Technical Details"):
                st.write(f"Label: {result['label']}")
                st.write(f"Probabilities: {result['probabilities']}")
                if hasattr(classifier, 'classes_'):
                    st.write(f"Model Classes: {classifier.classes_}")
        else:
            st.warning("Please enter some text to analyze")

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()