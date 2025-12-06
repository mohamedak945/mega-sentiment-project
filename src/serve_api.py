# serve_api.py - SIMPLIFIED
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import uvicorn

app = FastAPI(title="Sentiment Analysis API")

print("Loading sentiment analysis model...")

# 1. Load config
with open('artifacts/models/best_model_info.json', 'r') as f:
    config = json.load(f)

print(f"Model: {config['classifier']} on {config['embedding_model']}")
print(f"🎯 Accuracy: {config['metrics']['accuracy']:.4f}")

# 2. Load classifier (SIMPLIFIED - 3 lines!)
model_path = f"artifacts/models/{config['embedding_model']}_{config['classifier']}.joblib"
classifier = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# 3. Embedding function
def get_embedding(text: str, model_name: str = 'xlm-roberta-base'):
    """Convert text to embedding vector"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

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

# 4. Request/Response models
class TextRequest(BaseModel):
    text: str

# 5. API endpoints
@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API",
        "model": f"{config['classifier']} on {config['embedding_model']}",
        "accuracy": config['metrics']['accuracy']
    }

@app.post("/predict")
def predict(request: TextRequest):
    try:
        # Map embedding names
        model_mapping = {
            'bert': 'bert-base-uncased',
            'distilbert': 'distilbert-base-uncased',
            'xlmr': 'xlm-roberta-base'
        }
        
        embedding_model = model_mapping.get(config['embedding_model'], 'xlm-roberta-base')
        
        # Generate embedding
        embedding = get_embedding(request.text, embedding_model)
        
        # Make prediction
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        confidence = float(max(probabilities))
        
        sentiment = "positive" if prediction == 1 else "negative"
        
        return {
            "text": request.text,
            "sentiment": sentiment,
            "confidence": confidence,
            "label": int(prediction)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*50)
    print("API Ready at http://localhost:8000")
    print("Test with: POST /predict with JSON {\"text\": \"your text\"}")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)




    


    