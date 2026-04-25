# custom_handler_torchserve_fallback.py
from ts.torch_handler.base_handler import BaseHandler
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformersHandler(BaseHandler):
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_dir = os.getenv("MODEL_DIR", "/home/model-server/model")  # where torchserve will put model files
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Expect list of dicts or strings
        texts = []
        for item in data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or next(iter(item.values()), "")
            else:
                text = item
            texts.append(str(text))
        return texts

    def inference(self, inputs):
        enc = self.tokenizer(texts := inputs, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().tolist()
        return probs

    def postprocess(self, inference_output):
        labels = ["Negative","Neutral","Positive"]
        out = []
        for p in inference_output:
            idx = int(max(range(len(p)), key=lambda i: p[i]))
            out.append({"label": labels[idx], "confidence": float(p[idx])})
        return out
