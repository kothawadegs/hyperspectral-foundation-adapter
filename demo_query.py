import numpy as np
import torch
from models.hyper_vit import HyperDINOv2LoRA
from search.indexer import SpectralVectorIndex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HyperDINOv2LoRA(in_bands=204, num_classes=6).to(device)
model.eval()

# Simulate embedding extraction & vector indexing
vector_store = SpectralVectorIndex(dim=384)
mock_minerals = ["Alunite", "Kaolinite", "Calcite", "Buddingtonite", "Muscovite", "Jarosite"]

# Populate index with mock patches
np.random.seed(42)
mock_embeddings = np.random.randn(200, 384).astype(np.float32)
mock_metadata = [
    {
        "patch_id": i,
        "mineral": mock_minerals[i % len(mock_minerals)],
        "coordinates": f"{-21.11 + i*0.001:.4f}S, {119.73 + i*0.001:.4f}E (Pilbara, WA)"
    }
    for i in range(200)
]
vector_store.add_embeddings(mock_embeddings, mock_metadata)

# Query demo
target_query_vec = mock_embeddings[12:13].copy() # Alunite vector
hits = vector_store.query(target_query_vec, top_k=3)

print("\n--- Top Retrieved Satellite Targets ---")
for rank, hit in enumerate(hits, 1):
    m = hit["metadata"]
    print(f"Rank {rank}: Cosine Sim = {hit['score']:.4f} | Mineral: {m['mineral']} | Coords: {m['coordinates']}")
