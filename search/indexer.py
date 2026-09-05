import faiss
import numpy as np
import torch

class SpectralVectorIndex:
    """
    FAISS-backed spatial-spectral index with cosine similarity search.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, meta_list: list):
        # Normalize for Inner Product (equivalent to Cosine Similarity)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(meta_list)

    def query(self, query_vec: np.ndarray, top_k: int = 3):
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "score": float(dist),
                "metadata": self.metadata[idx]
            })
        return results
