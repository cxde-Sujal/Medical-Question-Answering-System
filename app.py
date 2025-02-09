import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Load Sentence Transformer Model for FAISS
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ✅ Load FAISS Index
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("medical_index.faiss")  # Load FAISS index
    with open("corpus.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip().lower() for line in f.readlines()]  # Load corpus
    return index, corpus

index, corpus = load_faiss_index()

# ✅ Load FLAN-T5 Model
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    return tokenizer, model, device

tokenizer, model, device = load_llm()

# ✅ FAISS Search Function
def search_dense(query, top_k=3):
    """Retrieve relevant contexts using FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # ✅ Filter out short, irrelevant text
    results = [
        corpus[idx] for i, idx in enumerate(indices[0]) 
        if idx < len(corpus) and distances[0][i] < 0.8 and len(corpus[idx].split()) > 5
    ]
    return results if results else ["No relevant context found."]

# ✅ Generate Medical Answer
def generate_answer(query):
    """Generate medical answers using FLAN-T5 with FAISS-retrieved context."""
    context = search_dense(query, top_k=3)
    context_text = " ".join(context)

    if "No relevant context found." in context_text:
        return "I'm sorry, but I couldn't find relevant information."

    input_text = f"""
    You are a medical assistant. Answer based on the given context.

    Context:
    {context_text}

    Question: {query}
    
    Provide a short, precise medical answer.
    Answer:
    """

    # Tokenize & Generate response
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    output = model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9, do_sample=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Streamlit UI
st.title("🩺 Medical Question Answering System")
query = st.text_input("🔎 Ask a medical question:")

if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)

    st.subheader("🤖 AI-Generated Answer:")
    st.write(answer)

    st.subheader("📚 Retrieved Contexts:")
    st.write(search_dense(query))

# ✅ Optional: Clear CUDA memory after processing
if device == "cuda":
    torch.cuda.empty_cache()
