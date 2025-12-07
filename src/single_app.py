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
    layout="wide"
)

# ============================================
# LOAD MODELS (EXACTLY LIKE YOUR API)
# ============================================

@st.cache_resource
def load_models():
    """Load all models exactly like your API does"""
    print("Loading sentiment analysis model...")
    
    # 1. Load config
    with open('artifacts/models/best_model_info.json', 'r') as f:
        config = json.load(f)
    
    print(f"Model: {config['classifier']} on {config['embedding_model']}")
    print(f"🎯 Accuracy: {config['metrics']['accuracy']:.4f}")
    
    # 2. Load classifier
    model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
    classifier = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return classifier, config

@st.cache_resource
def get_embedding_model():
    """Get embedding model and tokenizer"""
    config = load_models()[1]
    
    # Map embedding names (EXACTLY like your API)
    model_mapping = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'xlmr': 'xlm-roberta-base'
    }
    
    embedding_model_name = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
    
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    
    # Device selection (EXACTLY like your API)
    def select_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU Nvidia cuda will be our device")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Apple Silicon Chip will be our device")
        else:
            device = torch.device("cpu")
            print("CPU will be our device")
        return device
    
    device = select_device()
    model.to(device)
    model.eval()
    
    return tokenizer, model, device, embedding_model_name

# ============================================
# EMBEDDING FUNCTION (EXACTLY LIKE YOUR API)
# ============================================

def get_embedding(text: str, tokenizer, model, device):
    """Convert text to embedding vector - EXACT copy from your API"""
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
# PREDICTION FUNCTION (EXACTLY LIKE YOUR API)
# ============================================

def predict_sentiment(text: str):
    """Make prediction exactly like your API /predict endpoint"""
    try:
        # Load models
        classifier, config = load_models()
        tokenizer, model, device, embedding_model_name = get_embedding_model()
        
        # Generate embedding
        embedding = get_embedding(text, tokenizer, model, device)
        
        # Make prediction
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        confidence = float(max(probabilities))
        
        # DEBUG: Show raw values
        print(f"DEBUG - Text: {text}")
        print(f"DEBUG - Prediction: {prediction}")
        print(f"DEBUG - Probabilities: {probabilities}")
        print(f"DEBUG - Confidence: {confidence}")
        
        # Check if classifier has classes_ attribute
        if hasattr(classifier, 'classes_'):
            print(f"DEBUG - Classifier classes: {classifier.classes_}")
            print(f"DEBUG - Prediction maps to: {classifier.classes_[prediction]}")
        
        # THIS IS THE CRITICAL LINE FROM YOUR API
        sentiment = "positive" if prediction == 1 else "negative"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "label": int(prediction),
            "probabilities": probabilities.tolist()
        }
        
    except Exception as e:
        return {"error": str(e)}

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🧠 Sentiment Analysis")
    
    # Initialize session state
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
                if st.button("Positive Example", use_container_width=True, key="btn_positive"):
                    st.session_state.text_input = "This product is absolutely amazing! I've never been happier with a purchase."
                    st.rerun()
            
            with col_ex2:
                if st.button("Negative Example", use_container_width=True, key="btn_negative"):
                    st.session_state.text_input = "I'm very disappointed with this service. The quality is poor."
                    st.rerun()
    
    with col_right:
        # RESULTS DISPLAY
        st.subheader("Results")
        
        if analyze_button and text_input.strip():
            try:
                with st.spinner("Analyzing sentiment..."):
                    # Get prediction (EXACTLY like your API)
                    result = predict_sentiment(text_input)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        st.stop()
                
                # DISPLAY RESULTS
                if result["sentiment"] == "positive":
                    st.success(f"✅ POSITIVE SENTIMENT")
                else:
                    st.error(f"❌ NEGATIVE SENTIMENT")
                
                st.metric("Confidence", f"{result['confidence']:.2%}")
                st.metric("Label", result["label"])
                
                # Show probabilities
                if "probabilities" in result and len(result["probabilities"]) == 2:
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("P(Negative)", f"{result['probabilities'][0]:.2%}")
                    with col_prob2:
                        st.metric("P(Positive)", f"{result['probabilities'][1]:.2%}")
                
                # DEBUG INFO
                with st.expander("🔧 Debug Info"):
                    st.write(f"**Raw prediction label:** {result['label']}")
                    st.write(f"**Probabilities:** {result.get('probabilities', [])}")
                    st.write(f"**Interpreted sentiment:** {result['sentiment']}")
                    
                    # Load classifier to show classes
                    classifier, config = load_models()
                    if hasattr(classifier, 'classes_'):
                        st.write(f"**Model classes:** {classifier.classes_}")
                        st.write(f"**Prediction maps to class:** {classifier.classes_[result['label']]}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            st.info("Enter text and click 'Analyze Sentiment'")

# ============================================
# TEST FUNCTION - ADD THIS
# ============================================

def test_model():
    """Test the model with example texts"""
    st.sidebar.subheader("Model Test")
    
    if st.sidebar.button("Run Test"):
        test_texts = [
            ("I love this product! It's amazing!", "positive"),
            ("This is terrible, worst product ever.", "negative"),
            ("It's okay, nothing special.", "neutral")
        ]
        
        results = []
        for text, expected in test_texts:
            result = predict_sentiment(text)
            results.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "prediction": result.get("label", -1),
                "sentiment": result.get("sentiment", "error"),
                "expected": expected
            })
        
        # Display results
        import pandas as pd
        df = pd.DataFrame(results)
        st.sidebar.dataframe(df)
        
        # Check if predictions match expectations
        correct = sum(1 for r in results if r["sentiment"] == r["expected"])
        st.sidebar.write(f"Correct: {correct}/{len(results)}")

# ============================================
# APP ENTRY POINT
# ============================================
if __name__ == "__main__":
    # Add test function to sidebar
    with st.sidebar:
        test_model()
    
    main()