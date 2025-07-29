
# 🔍 RAG Chatbot – Retrieval-Augmented Generation Chatbot with ChromaDB + HuggingFace + Gradio

Welcome to the **RAG Chatbot**, a powerful, fully open-source conversational AI system that can answer your questions using your **own documents**!

Built with:
- 🧠 **LangChain** for RAG pipeline logic
- 📚 **ChromaDB** for local document storage & retrieval
- ✨ **Hugging Face Transformers** for LLMs (like FLAN-T5)
- 💬 **Gradio** for a beautiful user interface
- ☁️ **Hosted on Hugging Face Spaces**

---

## 🚀 Features

- 📄 Upload documents (PDF, TXT, CSV, DOCX)
- 🔍 Ask questions and get context-aware answers
- 💾 Uses **ChromaDB** to store and search chunks
- 🧠 Uses **FLAN-T5** or similar open-source models for generation
- 🌐 Works fully online — no local setup required
- 🔗 Shareable public URL with Hugging Face Spaces

---

## 📦 How It Works

1. **Document Upload**: Upload files from your device
2. **Vector Store Creation**: Text chunks are embedded and stored in ChromaDB
3. **Query Answering**: Your query is converted into a vector → similar chunks are retrieved
4. **LLM Generation**: Retrieved context is passed to the LLM (like `flan-t5-base`) to generate a response
5. **Response Displayed** in the Gradio UI

---

## 🧪 Try It Out

✅ Hosted version: https://huggingface.co/spaces/YasirUsman/rag-chatbot

---

## 🧰 Requirements

If you're running locally:

```bash
pip install -r requirements.txt
sudo apt-get install libmagic1
```
Or use this full setup in Colab:
```

!pip install -U langchain langchain-community huggingface_hub chromadb unstructured pypdf python-docx
```
