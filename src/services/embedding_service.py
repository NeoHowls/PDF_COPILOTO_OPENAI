
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings


class EmbeddingService:
    
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
    
    def encode_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
