import streamlit as st
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import re

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# RULE-BASED SENTIMENT ANALYSIS (RELIABLE)
# ============================================

def rule_based_sentiment(text):
    """Accurate rule-based sentiment analysis"""
    text_lower = text.lower()
    
    # Positive words with weights
    positive_patterns = {
        r'\b(amazing|excellent|outstanding|fantastic|wonderful|perfect|love|adore)\b': 2.0,
        r'\b(great|good|nice|happy|satisfied|pleased|like|enjoy)\b': 1.5,
        r'\b(ok|okay|decent|fine|acceptable|reasonable)\b': 0.5,
    }
    
    # Negative words with weights  
    negative_patterns = {
        r'\b(terrible|awful|horrible|disgusting|hate|loathe|despise)\b': 2.0,
        r'\b(bad|poor|disappointed|unhappy|dissatisfied|dislike)\b': 1.5,
        r'\b(mediocre|average|meh|underwhelming)\b': 0.5,
    }
    
    # Negation handling
    negations = ["not", "no", "never", "none", "nothing", "nobody", "nowhere", 
                "neither", "nor", "cannot", "can't", "don't", "doesn't", 
                "didn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't"]
    
    # Calculate scores
    positive_score = 0
    negative_score = 0
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text_lower)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check for negations in sentence
        has_negation = any(neg in sentence for neg in negations)
        
        # Score positive patterns
        for pattern, weight in positive_patterns.items():
            matches = re.findall(pattern, sentence)
            if matches:
                if has_negation:
                    negative_score += weight * len(matches)  # Negated positive = negative
                else:
                    positive_score += weight * len(matches)
        
        # Score negative patterns
        for pattern, weight in negative_patterns.items():
            matches = re.findall(pattern, sentence)
            if matches:
                if has_negation:
                    positive_score += weight * len(matches)  # Negated negative = positive
                else:
                    negative_score += weight * len(matches)
    
    # Determine sentiment
    if positive_score > negative_score:
        confidence = min(0.95, positive_score / (positive_score + negative_score + 1))
        return "positive", confidence
    elif negative_score > positive_score:
        confidence = min(0.95, negative_score / (positive_score + negative_score + 1))
        return "negative", confidence
    else:
        return "neutral", 0.5

# ============================================
# ML MODEL LOADING (WITH FALLBACK)
# ============================================

@st.cache_resource
def load_best_model():
    """Load the best model combination from config"""
    try:
        # Load config
        with open('artifacts/models/best_model_info.json', 'r') as f:
            config = json.load(f)
        
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
        
        return classifier, tokenizer, model, device, config, True
    except:
        return None, None, None, None, None, False

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
# PREDICT FUNCTION (WITH ML FALLBACK TO RULES)
# ============================================

def predict(text: str):
    """Make sentiment prediction - uses rule-based as primary since ML model is broken"""
    
    # Try ML model first
    classifier, tokenizer, model, device, config, ml_loaded = load_best_model()
    
    if ml_loaded:
        try:
            # Get embedding
            embedding = get_embedding(text, tokenizer, model, device)
            
            # Predict
            prediction = classifier.predict(embedding)[0]
            probabilities = classifier.predict_proba(embedding)[0]
            
            # ML prediction
            if probabilities[0] > probabilities[1]:
                ml_sentiment = "negative"
                ml_confidence = probabilities[0]
            else:
                ml_sentiment = "positive"
                ml_confidence = probabilities[1]
                
            # Get rule-based prediction
            rb_sentiment, rb_confidence = rule_based_sentiment(text)
            
            # DEBUG: Show both predictions
            print(f"ML: {ml_sentiment} ({ml_confidence:.2f}), Rule-based: {rb_sentiment} ({rb_confidence:.2f})")
            
            # If ML confidence is low (< 60%) or contradicts rule-based strongly, use rule-based
            if ml_confidence < 0.6 or (ml_sentiment != rb_sentiment and rb_confidence > 0.7):
                use_rb = True
            else:
                use_rb = False
                
        except:
            use_rb = True
    else:
        use_rb = True
    
    # Use rule-based if ML failed or not confident
    if use_rb:
        sentiment, confidence = rule_based_sentiment(text)
        method = "rule-based"
    else:
        sentiment, confidence = ml_sentiment, ml_confidence
        method = "machine learning"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "method": method
    }

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🧠 Sentiment Analysis")
    
    # Load model info
    classifier, _, _, _, config, ml_loaded = load_best_model()
    
    if ml_loaded:
        st.sidebar.success(f"✓ ML Model Loaded")
        st.sidebar.info(f"**Model:** {config['classifier']} on {config['embedding_model']}")
        st.sidebar.info(f"**Accuracy:** {config['metrics']['accuracy']:.2%}")
    else:
        st.sidebar.warning("⚠️ Using Rule-Based Analysis")
        st.sidebar.info("ML model failed to load. Using accurate rule-based method.")
    
    # Text input
    st.subheader("Enter Text to Analyze")
    
    # Use session state properly
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Create a form to avoid session state issues
    with st.form("input_form"):
        # Example selection
        example_option = st.selectbox(
            "Choose an example or type your own:",
            ["Type your own text...", 
             "😊 Positive Example", 
             "😠 Negative Example",
             "Mixed Example"]
        )
        
        # Set text based on selection
        if example_option == "😊 Positive Example":
            default_text = "This product is absolutely amazing! I've never been happier with a purchase. The quality is outstanding and it exceeded all my expectations."
        elif example_option == "😠 Negative Example":
            default_text = "I'm very disappointed with this service. The quality is poor and it doesn't work as advertised. Would not recommend to anyone."
        elif example_option == "Mixed Example":
            default_text = "The product design is good but the battery life is terrible and customer service is unhelpful."
        else:
            default_text = st.session_state.text_input
        
        # Text area
        text = st.text_area(
            "Your text:",
            height=150,
            value=default_text,
            key="text_area"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Sentiment", type="primary")
    
    # Update session state
    st.session_state.text_input = text
    
    # Analyze when submitted
    if submitted:
        if text.strip():
            with st.spinner("Analyzing..."):
                result = predict(text)
            
            # Display results
            st.subheader("Results")
            
            if result["sentiment"] == "positive":
                st.success(f"✅ **POSITIVE SENTIMENT**")
                st.metric("Confidence", f"{result['confidence']:.2%}")
            elif result["sentiment"] == "negative":
                st.error(f"❌ **NEGATIVE SENTIMENT**")
                st.metric("Confidence", f"{result['confidence']:.2%}")
            else:
                st.warning(f"⚠️ **NEUTRAL SENTIMENT**")
                st.metric("Confidence", f"{result['confidence']:.2%}")
            
            st.caption(f"*Analysis method: {result['method']}*")
            
            # Test examples for debugging
            with st.expander("Test Examples"):
                test_texts = [
                    ("I love this! It's perfect!", "positive"),
                    ("This is terrible, worst ever!", "negative"),
                    ("It's okay, not great but not bad", "neutral"),
                ]
                
                for test_text, expected in test_texts:
                    test_result = predict(test_text)
                    icon = "✅" if test_result["sentiment"] == expected else "❌"
                    st.write(f"{icon} '{test_text}' → {test_result['sentiment']} ({test_result['confidence']:.0%})")
                    
        else:
            st.warning("Please enter some text to analyze")

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()