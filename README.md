# Medical Question Answering System

## 📌 Project Overview
The **Medical Question Answering System** is an AI-powered application that provides accurate medical responses based on a pre-built knowledge base. It utilizes **FAISS for efficient retrieval** and **FLAN-T5 for natural language generation** to answer healthcare-related queries.

---

## 🚀 Deployment Instructions

### **1. Clone the Repository**
```sh
git clone https://github.com/cxde-Sujal/Medical-Question-Answering-System.git
cd Medical-Question-Answering-System
```

### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3. Set Up FAISS Index**
Since large files (like `medical_index.faiss`) are not in GitHub, download them separately and place them in the project directory.
```sh
python setup_faiss.py  # This script downloads the FAISS index
```

### **4. Run the Streamlit App**
```sh
streamlit run app.py
```

---

## 📖 Usage Guidelines

1. **Enter a medical question** in the Streamlit UI.
2. The system retrieves relevant contexts from the knowledge base using **FAISS**.
3. The **FLAN-T5 model** generates an answer based on retrieved contexts.
4. The output is displayed in an easy-to-read format.

---

## ⚙️ Technical Implementation

### **1. Data Processing**
- Medical textbooks and articles are converted into embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- FAISS stores and retrieves these embeddings efficiently.

### **2. Retrieval Using FAISS**
- The user query is converted into an embedding.
- FAISS finds the top `k` most relevant contexts.
- A similarity threshold is applied to filter results.

### **3. Response Generation Using FLAN-T5**
- The retrieved context is formatted as an input prompt.
- `FLAN-T5-small` generates a concise, factual response.
- Parameters like `temperature` and `top_p` control response variability.

---

## 🛠 Future Improvements
- ✅ Expand the knowledge base with more datasets.
- ✅ Improve retrieval accuracy with hybrid search (dense + sparse retrieval).
- ✅ Upgrade to a more powerful LLM (e.g., **Mistral-7B** if GPU available).

---

## 🤝 Contributors
- **Sujal Sinha** ([GitHub](https://github.com/cxde-Sujal))

---

## 📜 License
This project is licensed under the MIT License.

