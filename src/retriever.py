import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import json

class Retriever:
    def __init__(self, embedder_model_name: str = "all-MiniLM-L6-v2", cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Инициализация ретривера с моделью эмбеддингов и кросс-энкодером.
        """
        self.embedder = SentenceTransformer(embedder_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.index = None
        self.chunks = []  # список словарей {page_content, metadata}

    def build_index(self, chunks: List[Dict], batch_size: int = 32):
        """
        Строит индекс FAISS по эмбеддингам чанков.
        chunks: список словарей {page_content: str, metadata: Dict}.
        """
        if not chunks:
            raise ValueError("Список чанков пуст")

        self.chunks = chunks
        texts = [chunk["page_content"] for chunk in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"Индекс построен, добавлено {len(chunks)} чанков.")

    def search(self, query: str, top_k: int = 3, max_distance: float = 1.0) -> List[Tuple[Dict, float]]:
        """
        Поиск релевантных чанков по запросу.
        Возвращает список (metadata чанка, расстояние).
        """
        if self.index is None:
            raise RuntimeError("Индекс не построен")

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist > max_distance:
                continue
            results.append((self.chunks[idx]["metadata"], dist))
        return results

    def rerank(self, results: List[Tuple[Dict, float]], query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Переранжирование результатов с помощью кросс-энкодера.
        results: список (metadata, расстояние) из search.
        Возвращает отсортированный список (metadata, score).
        """
        if not results:
            return []

        pairs = [(query, chunk["page_content"]) for chunk, _ in results]
        scores = self.cross_encoder.predict(pairs)
        ranked = [(results[i][0], scores[i]) for i in range(len(results))]
        ranked.sort(key=lambda x: x[1], reverse=True)  # Высокий score — более релевантно
        return ranked[:top_k]

if __name__ == "__main__":
    # Загрузка данных
    with open("data/catalog_chunks.json", "r", encoding="utf-8") as f:
        catalog_chunks = json.load(f)
    with open("data/sales_templates.json", "r", encoding="utf-8") as f:
        sales_templates = json.load(f)

    retriever = Retriever()
    retriever.build_index(catalog_chunks + sales_templates)

    query = "моторчик омывателя для Golf 6"
    results = retriever.search(query, top_k=5)
    ranked_results = retriever.rerank(results, query, top_k=3)

    print("Результаты поиска:")
    for metadata, score in ranked_results:
        if "артикул" in metadata:
            print(f"- Запчасть: {metadata['название']} ({metadata['артикул']}), "
                  f"цена: {metadata['цена']} ₽, score={score:.4f}")
        elif "phrases" in metadata:
            print(f"- Фраза: {metadata['phrases'][0]}, situation={metadata['situation']}, score={score:.4f}")
