# serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vertex_model_server")

app = FastAPI(title="DistilBERT Sentiment Server")

# Expect model files to be present under MODEL_DIR (packaged into container)
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")  # we will copy model into /app/model in Dockerfile
LABELS = ["Negative", "Neutral", "Positive"]  # ensure the same ordering used during training

# Pydantic models for request
class InstancesRequest(BaseModel):
    instances: List[Any]  # Accept a list of strings or dicts with "text"/"content" keys

class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]

# lazy load
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    logger.info(f"Loading model from {MODEL_DIR} on device {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model from MODEL_DIR. Ensure model files were copied into the container.")
        raise

def extract_text_from_instance(instance):
    # instance can be a plain string or a dict with keys "text" or "content"
    if isinstance(instance, str):
        return instance
    if isinstance(instance, dict):
        for key in ("text", "content", "input", "sentence"):
            if key in instance:
                return instance[key]
    return str(instance)

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error("Model failed to load during startup: %s", e)

@app.post("/predict", response_model=PredictResponse)
def predict(req: InstancesRequest):
    """
    Expects:
    {
      "instances": [
         "I love this!",
         {"text":"Not good at all"}
      ]
    }

    Returns:
    {
      "predictions": [
        {"label":"Positive", "confidence":0.95},
        {"label":"Negative", "confidence":0.78}
      ]
    }
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    texts = [extract_text_from_instance(x) for x in req.instances]
    if len(texts) == 0:
        return {"predictions": []}

    # Tokenize with padding/truncation
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    preds = []
    for p in probs:
        idx = int(p.argmax())
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        conf = float(p[idx])
        preds.append({"label": label, "confidence": conf})

    return {"predictions": preds}
