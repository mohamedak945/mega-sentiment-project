# single_app.py - All-in-one Streamlit app
import streamlit as st
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    try:
        # Load config
        with open('artifacts/models/best_model_info.json', 'r') as f:
            config = json.load(f)
        
        # Load classifier
        model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
        classifier = joblib.load(model_path)
        
        st.success(f"✅ Model loaded: {config['classifier']} on {config['embedding_model']}")
        return config, classifier
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

@st.cache_resource
def get_embedding_model(model_name='xlm-roberta-base'):
    """Load transformer model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
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

def predict_sentiment(text, tokenizer, model, device, classifier):
    """Predict sentiment for a single text"""
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    embedding_np = embedding.cpu().numpy()
    
    # Predict
    prediction = classifier.predict(embedding_np)[0]
    probabilities = classifier.predict_proba(embedding_np)[0]
    confidence = float(max(probabilities))
    
    sentiment = "positive" if prediction == 1 else "negative"
    
    return sentiment, confidence, prediction

# Main app
def main():
    st.title("Sentiment Analyzer")
    st.markdown("---")
    
    # Model mapping
    model_mapping = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'xlmr': 'xlm-roberta-base'
    }
    
    # Load models
    with st.spinner("Loading models..."):
        config, classifier = load_models()
        
    if config is None or classifier is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Embedding model name
    embedding_name = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
    
    # Load transformer model
    with st.spinner(f"Loading {embedding_name}..."):
        tokenizer, model, device = get_embedding_model(embedding_name)
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Model Info")
        st.write(f"**Classifier:** {config['classifier']}")
        st.write(f"**Embedding Model:** {config['embedding_model']}")
        st.write(f"**Accuracy:** {config['metrics']['accuracy']:.4f}")
        st.write(f"**Device:** {device}")
        st.markdown("---")
        
        # Examples
        st.subheader("📝 Try These Examples:")
        examples = [
            "I absolutely love this product!",
            "This is the worst experience ever.",
            "The service was okay, nothing special.",
            "Amazing customer support!",
            "Terrible quality, would not recommend."
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example[:10]}"):
                st.session_state.example_text = example
    
    # Text input
    text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get('example_text', 'I love this product!'),
        height=100
    )
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if text.strip():
            with st.spinner("Analyzing..."):
                try:
                    sentiment, confidence, label = predict_sentiment(
                        text, tokenizer, model, device, classifier
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "positive":
                            st.success(f"😊 **POSITIVE**")
                        else:
                            st.error(f"😞 **NEGATIVE**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Progress bar
                    st.progress(confidence)
                    
                    # Additional info
                    with st.expander("📋 Details"):
                        st.write(f"**Text:** {text}")
                        st.write(f"**Sentiment:** {sentiment}")
                        st.write(f"**Label:** {label}")
                        st.write(f"**Confidence Score:** {confidence:.4f}")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Batch analysis
    st.markdown("---")
    st.subheader("📁 Batch Analysis")
    
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=150,
        placeholder="I love it!\nThis is terrible.\nIt's okay."
    )
    
    if st.button("📊 Analyze Batch"):
        if batch_text.strip():
            texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, text_line in enumerate(texts):
                try:
                    sentiment, confidence, label = predict_sentiment(
                        text_line, tokenizer, model, device, classifier
                    )
                    results.append({
                        "Text": text_line[:50] + "..." if len(text_line) > 50 else text_line,
                        "Sentiment": sentiment,
                        "Confidence": f"{confidence:.2%}",
                        "Label": label
                    })
                except:
                    results.append({
                        "Text": text_line[:50] + "...",
                        "Sentiment": "Error",
                        "Confidence": "N/A",
                        "Label": "N/A"
                    })
                
                progress_bar.progress((i + 1) / len(texts))
            
            # Display results
            st.dataframe(results)
            
            # Summary
            positive_count = sum(1 for r in results if r["Sentiment"] == "positive")
            st.info(f"📈 **Summary:** {positive_count} positive, {len(results)-positive_count} negative")
        else:
            st.warning("Please enter some text for batch analysis.")

if __name__ == "__main__":
    main()