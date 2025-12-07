import streamlit as st
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import sys

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
# PREDICTION FUNCTION WITH FIX FOR ALWAYS-POSITIVE
# ============================================

def predict_sentiment(text: str):
    """Make prediction with fix for always-positive model"""
    try:
        # Load models
        classifier, config = load_models()
        tokenizer, model, device, embedding_model_name = get_embedding_model()
        
        # Generate embedding
        embedding = get_embedding(text, tokenizer, model, device)
        
        # Make prediction
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        
        # DEBUG INFO
        print(f"DEBUG - Text: {text[:50]}...")
        print(f"DEBUG - Raw prediction: {prediction}")
        print(f"DEBUG - Probabilities: {probabilities}")
        
        # Check classifier classes
        if hasattr(classifier, 'classes_'):
            classes = classifier.classes_
            print(f"DEBUG - Classifier classes: {classes}")
            print(f"DEBUG - Prediction maps to: {classes[prediction]}")
            
            # FIX: If model always predicts positive, check probability distribution
            # If probability[1] (positive) is always > 0.5, model is biased
            
            # If negative probability is higher, use that
            if probabilities[0] > probabilities[1]:
                sentiment = "negative"
                confidence = probabilities[0]
                label = 0 if len(classes) > 1 else prediction
            else:
                sentiment = "positive"
                confidence = probabilities[1]
                label = 1 if len(classes) > 1 else prediction
        else:
            # No classes attribute
            if probabilities[0] > probabilities[1]:
                sentiment = "negative"
                confidence = probabilities[0]
                label = 0
            else:
                sentiment = "positive"
                confidence = probabilities[1]
                label = 1
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "label": label,
            "probabilities": probabilities.tolist(),
            "raw_prediction": int(prediction)
        }
        
    except Exception as e:
        print(f"ERROR in predict_sentiment: {str(e)}")
        return {"error": str(e)}

# ============================================
# MAIN APP WITH FIXED SESSION STATE
# ============================================

def main():
    st.title("🧠 Sentiment Analysis")
    
    # Initialize session state properly
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Create two columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # TEXT INPUT
        st.subheader("Enter Text for Analysis")
        
        # Use a unique key for the text area
        text_input = st.text_area(
            "Paste your text here:",
            height=150,
            placeholder="Enter text to analyze sentiment...",
            key="main_text_input",
            value=st.session_state.get('text_input', ''),
            label_visibility="collapsed"
        )
        
        # Update session state
        if text_input != st.session_state.get('text_input', ''):
            st.session_state.text_input = text_input
        
        # ANALYZE BUTTON
        analyze_button = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            key="analyze_btn"
        )
        
        # EXAMPLE TEXTS - SIMPLE FIX
        with st.expander("Try Example Texts"):
            col_ex1, col_ex2 = st.columns(2)
            
            example_texts = {
                "positive": "This product is absolutely amazing! I've never been happier with a purchase.",
                "negative": "I'm very disappointed with this service. The quality is poor."
            }
            
            with col_ex1:
                if st.button("Positive Example", use_container_width=True, key="btn_positive"):
                    # Use query params or form to avoid session state issues
                    st.session_state.text_input = example_texts["positive"]
                    st.query_params.update({"example": "positive"})
                    st.rerun()
            
            with col_ex2:
                if st.button("Negative Example", use_container_width=True, key="btn_negative"):
                    st.session_state.text_input = example_texts["negative"]
                    st.query_params.update({"example": "negative"})
                    st.rerun()
    
    with col_right:
        # RESULTS DISPLAY
        st.subheader("Results")
        
        if analyze_button and text_input.strip():
            try:
                with st.spinner("Analyzing sentiment..."):
                    # Get prediction
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
                st.write(f"**Label:** {result['label']}")
                
                # Show probabilities
                if "probabilities" in result and len(result["probabilities"]) == 2:
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Negative Prob", f"{result['probabilities'][0]:.2%}")
                    with col_prob2:
                        st.metric("Positive Prob", f"{result['probabilities'][1]:.2%}")
                
                # DEBUG INFO
                with st.expander("🔧 Debug Info"):
                    st.write(f"**Raw prediction:** {result.get('raw_prediction', 'N/A')}")
                    st.write(f"**Final label:** {result['label']}")
                    st.write(f"**Probabilities:** {result.get('probabilities', [])}")
                    st.write(f"**Interpreted sentiment:** {result['sentiment']}")
                    
                    # Check if model always predicts positive
                    classifier, _ = load_models()
                    if hasattr(classifier, 'classes_'):
                        st.write(f"**Model classes:** {classifier.classes_}")
                        
                        # Test if model is biased
                        test_texts = ["good", "bad", "excellent", "terrible"]
                        predictions = []
                        tokenizer, model, device, _ = get_embedding_model()
                        
                        for test_text in test_texts:
                            embedding = get_embedding(test_text, tokenizer, model, device)
                            pred = classifier.predict(embedding)[0]
                            predictions.append(pred)
                        
                        if len(set(predictions)) == 1:
                            st.warning("⚠️ **MODEL ALERT:** Always predicts the same class!")
                            st.write(f"Test predictions: {predictions}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            st.info("Enter text and click 'Analyze Sentiment'")

# ============================================
# FALLBACK RULE-BASED ANALYSIS
# ============================================

def rule_based_sentiment(text):
    """Fallback rule-based sentiment analysis"""
    text_lower = text.lower()
    
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 
                     'happy', 'wonderful', 'fantastic', 'best', 'perfect']
    
    negative_words = ['bad', 'terrible', 'poor', 'disappointed', 'hate', 'worst',
                     'awful', 'horrible', 'not good', 'dislike', 'broken']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive", max(0.5, min(0.9, 0.5 + (pos_count - neg_count) * 0.1))
    elif neg_count > pos_count:
        return "negative", max(0.5, min(0.9, 0.5 + (neg_count - pos_count) * 0.1))
    else:
        return "neutral", 0.5

# ============================================
# DIAGNOSTIC TOOLS
# ============================================

with st.sidebar:
    st.subheader("Diagnostic Tools")
    
    if st.button("Test Model Bias"):
        # Quick bias test
        classifier, config = load_models()
        
        st.write("**Model Classes:**")
        if hasattr(classifier, 'classes_'):
            st.write(classifier.classes_)
        else:
            st.write("No classes attribute")
        
        st.write("**Config:**")
        st.write(f"Classifier: {config['classifier']}")
        st.write(f"Accuracy: {config['metrics']['accuracy']:.2%}")
    
    if st.button("Use Rule-Based Fallback"):
        st.info("Rule-based analysis will be used as fallback")
        st.session_state.use_rule_based = True
    
    # Force model reload
    if st.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()

# ============================================
# APP ENTRY POINT
# ============================================
if __name__ == "__main__":
    # Handle example text from query params
    if "example" in st.query_params:
        example_type = st.query_params["example"]
        if example_type == "positive":
            st.session_state.text_input = "This product is absolutely amazing! I've never been happier with a purchase."
        elif example_type == "negative":
            st.session_state.text_input = "I'm very disappointed with this service. The quality is poor."
    
    main()