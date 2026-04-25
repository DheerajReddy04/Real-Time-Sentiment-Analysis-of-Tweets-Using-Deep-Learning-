# # vertex_client.py
# from google.cloud import aiplatform
# from typing import List, Tuple

# class VertexSentimentClient:
#     def __init__(self, project_id: str, region: str, endpoint_id: str):
#         """
#         Initialize the Vertex AI client.
#         project_id  : your GCP project ID
#         region      : region where the model is deployed (e.g., "us-central1")
#         endpoint_id : deployed Vertex AI endpoint ID
#         """
#         self.project_id = project_id
#         self.region = region
#         self.endpoint_id = endpoint_id
#         self.client = aiplatform.gapic.PredictionServiceClient(client_options={
#             "api_endpoint": f"{region}-aiplatform.googleapis.com"
#         })
#         self.endpoint_path = self.client.endpoint_path(
#             project=project_id,
#             location=region,
#             endpoint=endpoint_id
#         )

#     def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
#         """
#         Send preprocessed texts to Vertex AI and return sentiment predictions.
#         Returns list of tuples: (predicted_label, confidence)
#         """
#         predictions = []
#         instances = [{"content": t} for t in texts]

#         try:
#             response = self.client.predict(endpoint=self.endpoint_path, instances=instances)
#             for pred in response.predictions:
#                 # Expecting the model to return {"label": str, "confidence": float}
#                 label = pred.get("label", "Neutral")
#                 conf = float(pred.get("confidence", 0.6))
#                 predictions.append((label, conf))
#         except Exception as e:
#             print(f"[VertexAI ERROR] {e}")
#             # fallback: neutral with medium confidence
#             predictions = [("Neutral", 0.6) for _ in texts]

#         return predictions








# vertex_client.py
from google.cloud import aiplatform
from google.api_core.client_options import ClientOptions
from typing import List, Tuple
import os

class VertexSentimentClient:
    def __init__(self, project_id: str, region: str, endpoint_id: str):
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        api_endpoint = f"{region}-aiplatform.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        self.endpoint = self.client.endpoint_path(project=project_id, location=region, endpoint=endpoint_id)

    def predict(self, texts: List[str]) -> List[Tuple[str,float]]:
        instances = []
        for t in texts:
            # match the server's expected input format (see serve.py)
            instances.append({"text": t})
        try:
            response = self.client.predict(endpoint=self.endpoint, instances=instances)
            preds = []
            for pred in response.predictions:
                # server replies with {"label":..., "confidence":...}
                if isinstance(pred, dict):
                    label = pred.get("label", "Neutral")
                    conf = float(pred.get("confidence", 0.0))
                elif isinstance(pred, list) or isinstance(pred, (str, int, float)):
                    # fallback: unexpected
                    label = "Neutral"
                    conf = 0.0
                else:
                    label = pred.get("label","Neutral") if hasattr(pred, "get") else "Neutral"
                    conf = float(pred.get("confidence", 0.0)) if hasattr(pred, "get") else 0.0
                preds.append((label, conf))
            return preds
        except Exception as e:
            print("[Vertex AI ERROR]", e)
            # fallback neutral
            return [("Neutral", 0.6) for _ in texts]
