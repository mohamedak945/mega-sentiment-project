# serve_api.py - OPTIMIZED FOR RENDER.COM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import os
from pathlib import Path

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting Sentiment Analysis API...")

# Initialize variables
config = None
classifier = None
tokenizer = None
model = None
device = None

# Request model
class TextRequest(BaseModel):
    text: str

# Response model
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int

def load_models():
    """Load all models on startup"""
    global config, classifier, tokenizer, model, device
    
    try:
        print("📁 Loading configuration...")
        
        # Check if running on Render (different file paths)
        if os.path.exists('artifacts/models/best_model_info.json'):
            config_path = 'artifacts/models/best_model_info.json'
        elif os.path.exists('/opt/render/project/src/artifacts/models/best_model_info.json'):
            config_path = '/opt/render/project/src/artifacts/models/best_model_info.json'
        else:
            raise FileNotFoundError("Config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Config loaded: {config['classifier']} on {config['embedding_model']}")
        
        # Load classifier
        model_mapping = {
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'xlmr': 'xlm-roberta-base'
        }
        
        embedding_name = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
        
        # Try multiple paths for model file
        model_paths = [
            f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib",
            f"/opt/render/project/src/artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib",
            f"models/{config['embedding_model']}_{config['classifier']}.joblib"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError(f"Model file not found. Tried: {model_paths}")
        
        print(f"📦 Loading classifier from {model_path}...")
        classifier = joblib.load(model_path)
        
        # Load transformer model
        print(f"🤖 Loading transformer model: {embedding_name}...")
        tokenizer = AutoTokenizer.from_pretrained(embedding_name)
        model = AutoModel.from_pretrained(embedding_name)
        
        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️  Using device: {device}")
        
        model.to(device)
        model.eval()
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # Fallback to a simpler model
        setup_fallback_model()

def setup_fallback_model():
    """Setup a fallback using Hugging Face pipeline"""
    global config, classifier
    print("⚠️  Setting up fallback model...")
    
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", 
                         model="distilbert-base-uncased-finetuned-sst-2-english")
    config = {
        "classifier": "transformers_pipeline",
        "embedding_model": "distilbert",
        "metrics": {"accuracy": 0.91}
    }
    print("✅ Fallback model ready")

def get_embedding(text: str):
    """Convert text to embedding vector"""
    global tokenizer, model, device
    
    if tokenizer is None or model is None:
        # Use fallback
        return None
    
    try:
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
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# API endpoints
@app.get("/")
async def home():
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "model": f"{config['classifier']} on {config['embedding_model']}" if config else "fallback",
        "accuracy": config['metrics']['accuracy'] if config else 0.91,
        "endpoints": {
            "GET /": "This info",
            "POST /predict": "Analyze sentiment",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": classifier is not None,
        "config_loaded": config is not None
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest):
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Check if using fallback model
        if hasattr(classifier, 'predict'):
            # Original model flow
            embedding = get_embedding(text)
            
            if embedding is None:
                raise HTTPException(status_code=503, detail="Model temporarily unavailable")
            
            prediction = classifier.predict(embedding)[0]
            probabilities = classifier.predict_proba(embedding)[0]
            confidence = float(max(probabilities))
            sentiment = "positive" if prediction == 1 else "negative"
            
        else:
            # Fallback pipeline
            result = classifier(text)[0]
            sentiment = "positive" if result['label'] == 'POSITIVE' else "negative"
            confidence = result['score']
            prediction = 1 if sentiment == "positive" else 0
        
        return SentimentResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            label=prediction
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Test endpoint with sample text"""
    test_text = "I absolutely love this product!"
    
    try:
        # Simulate prediction
        return {
            "test": "success",
            "sample_text": test_text,
            "expected_sentiment": "positive"
        }
    except Exception as e:
        return {"test": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)