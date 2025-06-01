import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict
from tqdm import tqdm

class Retriever:
    def __init__(self, embedder_model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Индекс построен, добавлено {len(chunks)} чанков.")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        query_embedding = self.embedder.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [(self.chunks[idx]["metadata"], dist) for idx, dist in zip(indices[0], distances[0])]
        return results

    def rerank(self, results: List[Tuple[Dict, float]], query: str) -> List[Tuple[Dict, float]]:
        query_embedding = self.embedder.encode([query])[0]
        ranked = []
        for metadata, _ in results:
            chunk_text = next(chunk["text"] for chunk in self.chunks if chunk["metadata"] == metadata)
            chunk_embedding = self.embedder.encode([chunk_text])[0]
            score = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            ranked.append((metadata, score))
        return sorted(ranked, key=lambda x: x[1], reverse=True)