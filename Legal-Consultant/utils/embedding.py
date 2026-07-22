from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_db(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    os.makedirs("vector_db", exist_ok=True)

    faiss.write_index(index, "vector_db/legal.index")

    with open("vector_db/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector Database Created Successfully!")