import streamlit as st
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Load model once
@st.cache_resource
def load_model():
    with open('artifacts/models/best_model_info.json', 'r') as f:
        config = json.load(f)
    
    model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
    classifier = joblib.load(model_path)
    
    return config, classifier

@st.cache_resource
def load_embedding_model():
    model_mapping = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'xlmr': 'xlm-roberta-base'
    }
    
    config, _ = load_model()
    embedding_model_name = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
    
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

def get_embedding(text, tokenizer, model, device):
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

# Streamlit UI
st.title("Sentiment Analyzer 🎯")
st.write("Analyze text sentiment using ML models")

# Load models
config, classifier = load_model()
tokenizer, embedding_model, device = load_embedding_model()

st.sidebar.info(f"Model: {config['classifier']} on {config['embedding_model']}")
st.sidebar.info(f"Accuracy: {config['metrics']['accuracy']:.2%}")

# Text input
text = st.text_area("Enter text to analyze:", 
                    "I absolutely love this product! It's amazing!", 
                    height=100)

if st.button("Analyze Sentiment", type="primary"):
    with st.spinner("Analyzing..."):
        # Get embedding
        embedding = get_embedding(text, tokenizer, embedding_model, device)
        
        # Make prediction
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        confidence = float(max(probabilities))
        
        sentiment = "positive" if prediction == 1 else "negative"
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if sentiment == "positive":
                st.success(f"😊 **POSITIVE**")
            else:
                st.error(f"😞 **NEGATIVE**")
        
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Show details
        with st.expander("Details"):
            st.write(f"**Text:** {text}")
            st.write(f"**Prediction:** {sentiment} (label: {prediction})")
            st.write(f"**Model:** {config['classifier']} on {config['embedding_model']}")

st.divider()
st.caption("Built with Streamlit | Transformer models")