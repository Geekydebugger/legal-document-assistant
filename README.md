# ⚖️ Legal Document Assistant
### Agentic AI Capstone Project | Abhishek Chatterjee

An intelligent AI assistant that helps paralegals, junior lawyers, law students, and individuals understand legal clauses, contract types, and compute legal deadlines accurately.

---

## 🎯 Problem Statement

Legal professionals frequently need quick, reliable explanations of legal clauses and contract types. Consulting a senior lawyer for every basic question is time-consuming and expensive. Additionally, legal deadline calculations (notice periods, filing deadlines) are prone to human error when done manually.

---

## ✅ Mandatory Capabilities

| Capability | Status |
|-----------|--------|
| LangGraph StateGraph (8 nodes) | ✅ Implemented |
| ChromaDB RAG (12 documents) | ✅ Implemented |
| Conversation Memory (MemorySaver + thread_id) | ✅ Implemented |
| Self-Reflection Eval Node (faithfulness scoring) | ✅ Implemented |
| Tool Use (Date Calculator) | ✅ Implemented |
| Deployment (Streamlit UI) | ✅ Implemented |

---

## 🛠️ Tech Stack

- **LLM:** LLaMA 3.3 70B via Groq API
- **Agent Framework:** LangGraph
- **Vector Database:** ChromaDB
- **Embeddings:** SentenceTransformer (all-MiniLM-L6-v2)
- **Memory:** LangGraph MemorySaver
- **Tool:** Python datetime (date calculator)
- **UI:** Streamlit
- **Language:** Python 3.12

---

## 📄 Knowledge Base (12 Documents)

1. Non-Disclosure Agreement (NDA)
2. Employment Contract Terms and Termination
3. Rental and Lease Agreement
4. Service Agreement and Scope of Work
5. Intellectual Property Rights and Ownership
6. Dispute Resolution: Arbitration vs Litigation
7. Contract Termination Clauses and Notice Periods
8. Indemnification Clauses
9. Privacy Policy and Data Protection (GDPR)
10. Legal Disclaimer and Limitation of Liability
11. Contract Formation and Essential Elements
12. Non-Compete and Restraint of Trade Clauses

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/legal-document-assistant
cd legal-document-assistant
```

### 2. Install dependencies
```bash
pip install langgraph langchain-groq langchain-core chromadb sentence-transformers streamlit ragas datasets python-dotenv
```

### 3. Set up API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com)

### 4. Run the Streamlit UI
```bash
streamlit run capstone_streamlit.py
```

### 5. Run the notebook
Open `day13_capstone.ipynb` in Jupyter and run all cells.

---

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.520 |
| Answer Relevance | 0.850 |
| Context Precision | 0.800 |

**Test Results: 11/11 PASS | Red-team: 2/2 PASS**

---

## 🏗️ Agent Architecture

```
User Question
     ↓
[memory_node] → add to history, sliding window, extract name
     ↓
[router_node] → retrieve / memory_only / tool
     ↓
[retrieval_node / skip_node / tool_node]
     ↓
[answer_node] → grounded response from context only
     ↓
[eval_node] → faithfulness score → retry if < 0.7
     ↓
[save_node] → append to history → END
```

---

## 📁 Project Files

```
legal-document-assistant/
├── day13_capstone.ipynb          # Complete notebook (all 8 parts)
├── capstone_streamlit.py         # Streamlit web UI
├── agent.py                      # Standalone agent module
├── Capstone_Documentation_Abhishek_Chatterjee.pdf  # Project documentation
└── README.md                     # This file
```

---

## 🔮 Future Improvements

- Load real legal PDF documents using a PDF loader
- Implement hybrid BM25 + vector search for better precision
- Add multilingual support (Hindi, Bengali)
- Allow users to upload their own contracts
- Deploy on cloud with authentication

---

## ⚠️ Disclaimer

This assistant is for **informational and educational purposes only**. It does not constitute legal advice. Always consult a qualified legal professional for specific legal matters.

---

**Abhishek Chatterjee | Agentic AI Hands-On Course 2026 | Dr. Kanthi Kiran Sirra**
