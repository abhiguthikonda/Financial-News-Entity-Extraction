"""
Financial NER API - FastAPI Backend
Extracts financial entities from text using trained ML models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import json
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Financial NER API",
    description="""
    Named Entity Recognition API for Financial Text Analysis.
    
    Extracts key entities from financial news and reports:
    - **COMPANY**: Company names (Tesla, Apple, Microsoft)
    - **TICKER**: Stock symbols (AAPL, TSLA, MSFT)
    - **CURRENCY**: Currency mentions ($, USD, EUR)
    - **INDICATOR**: Financial metrics (revenue, profit, EPS)
    - **EVENT**: Financial events (IPO, merger, acquisition)
    """,
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
vectorizer = None
label_mapping = None
metadata = None

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_ner_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_PATH = os.path.join(MODEL_DIR, "label_mapping.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# Request/Response Models
class TextRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Tesla TSLA reported quarterly revenue of $25 billion."
            }
        }


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class TokenPrediction(BaseModel):
    token: str
    label: str


class PredictionResponse(BaseModel):
    text: str
    entities: List[Entity]
    tokens: List[TokenPrediction]


class BatchTextRequest(BaseModel):
    texts: List[str]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    labels: List[str]
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model():
    """Load the trained model and vectorizer."""
    global model, vectorizer, label_mapping, metadata
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}")
            return False
            
        if os.path.exists(VECTORIZER_PATH):
            vectorizer = joblib.load(VECTORIZER_PATH)
            print(f"✅ Vectorizer loaded from {VECTORIZER_PATH}")
        else:
            print(f"⚠️ Vectorizer not found at {VECTORIZER_PATH}")
            return False
            
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r') as f:
                label_mapping = json.load(f)
            print(f"✅ Label mapping loaded from {LABEL_PATH}")
            
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded from {METADATA_PATH}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def predict_entities(text: str) -> PredictionResponse:
    """
    Predict entities in the given text.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tokens = text.split()
    
    if not tokens:
        return PredictionResponse(text=text, entities=[], tokens=[])
    
    # Transform tokens using vectorizer
    X = vectorizer.transform(tokens)
    
    # Get predictions
    predictions = model.predict(X)
    
    # Build token predictions
    token_predictions = [
        TokenPrediction(token=token, label=label)
        for token, label in zip(tokens, predictions)
    ]
    
    # Extract entities with positions
    entities = []
    current_entity = None
    current_tokens = []
    current_start = 0
    char_pos = 0
    
    for i, (token, label) in enumerate(zip(tokens, predictions)):
        token_start = text.find(token, char_pos)
        token_end = token_start + len(token)
        
        if label.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entity_text = ' '.join(current_tokens)
                entities.append(Entity(
                    text=entity_text,
                    label=current_entity,
                    start=current_start,
                    end=char_pos - 1
                ))
            
            # Start new entity
            current_entity = label[2:]
            current_tokens = [token]
            current_start = token_start
            
        elif label.startswith('I-') and current_entity == label[2:]:
            # Continue current entity
            current_tokens.append(token)
            
        else:
            # End current entity
            if current_entity:
                entity_text = ' '.join(current_tokens)
                entities.append(Entity(
                    text=entity_text,
                    label=current_entity,
                    start=current_start,
                    end=char_pos - 1
                ))
                current_entity = None
                current_tokens = []
        
        char_pos = token_end + 1
    
    # Don't forget last entity
    if current_entity:
        entity_text = ' '.join(current_tokens)
        entities.append(Entity(
            text=entity_text,
            label=current_entity,
            start=current_start,
            end=len(text)
        ))
    
    return PredictionResponse(
        text=text,
        entities=entities,
        tokens=token_predictions
    )


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("\n" + "=" * 50)
    print("🚀 Starting Financial NER API")
    print("=" * 50)
    load_model()
    print("=" * 50 + "\n")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Financial NER API",
        "description": "Named Entity Recognition for Financial Text",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return ModelInfo(
        model_name=metadata.get("model_name", "Unknown"),
        accuracy=metadata.get("accuracy", 0.0),
        f1_macro=metadata.get("f1_macro", 0.0),
        f1_weighted=metadata.get("f1_weighted", 0.0),
        labels=label_mapping.get("labels", []) if label_mapping else [],
        status="loaded" if model is not None else "not loaded"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: TextRequest):
    """
    Extract financial entities from text.
    
    - **text**: The financial text to analyze
    
    Returns extracted entities with their types and positions.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return predict_entities(request.text)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchTextRequest):
    """
    Extract entities from multiple texts.
    
    - **texts**: List of financial texts to analyze
    
    Returns predictions for each text.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    predictions = []
    for text in request.texts:
        if text.strip():
            predictions.append(predict_entities(text))
    
    return BatchPredictionResponse(predictions=predictions)


@app.post("/reload-model", tags=["Model"])
async def reload_model():
    """Reload the model from disk."""
    success = load_model()
    if success:
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


# Entity colors for frontend
ENTITY_COLORS = {
    "COMPANY": "#ff6b6b",
    "TICKER": "#4ecdc4",
    "CURRENCY": "#45b7d1",
    "INDICATOR": "#96ceb4",
    "EVENT": "#feca57"
}


@app.get("/entity-colors", tags=["Utilities"])
async def get_entity_colors():
    """Get color codes for entity visualization."""
    return ENTITY_COLORS


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Financial NER API Server")
    print("=" * 50)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
