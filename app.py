import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load the embedding model only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ✅ Load FAISS index only once
@st.cache_resource
def load_faiss_index():
    # Load corpus
    with open("corpus.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip().lower() for line in f.readlines()]
    
    # Encode corpus
    embeddings = embedding_model.encode(corpus, convert_to_numpy=True)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, corpus

index, corpus = load_faiss_index()

# ✅ FAISS search function with error handling
def search_dense(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [corpus[idx] for i, idx in enumerate(indices[0]) if idx < len(corpus) and distances[0][i] < 0.8]
    
    return results if results else ["No relevant context found."]

# ✅ Example Streamlit UI
st.title("Medical Question Answering System")

query = st.text_input("Enter your medical question:")
if query:
    results = search_dense(query)
    st.write("Relevant Contexts:", results)
