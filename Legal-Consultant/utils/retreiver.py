import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector_db/legal.index")

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve(query, top_k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results