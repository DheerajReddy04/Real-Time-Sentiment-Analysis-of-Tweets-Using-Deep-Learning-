from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TransformersHandler:
    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert_tweeteval_boosted")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert_tweeteval_boosted")
        self.model.to(self.device)
        self.initialized = True

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)
        inputs = self.tokenizer(data[0].get("text"), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores.tolist()
