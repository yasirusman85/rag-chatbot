

# ✅ Imports
import os
import gradio as gr
import PyPDF2

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Load LLM (FLAN-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# ✅ Prepare persistent ChromaDB directory (if needed)
CHROMA_DIR = "chroma_db"
if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)

# ✅ Globals
db = None

# ✅ Chunking function
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# ✅ Extract text from file (PDF or TXT)
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        return file.read().decode("utf-8")

# ✅ Upload and index function
def process_file(file):
    global db
    text = extract_text(file)
    chunks = chunk_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create ChromaDB from documents
    db = Chroma.from_documents(documents, embedding=embedding_fn, persist_directory=CHROMA_DIR)
    db.persist()

    return f"✅ Successfully indexed {len(chunks)} chunks from your document."

# ✅ RAG chat function
def chat(query):
    global db
    if db is None:
        return "⚠️ Please upload and index a document first."

    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 RAG Chatbot using FLAN-T5 + ChromaDB")

    with gr.Row():
        file_input = gr.File(label="📄 Upload PDF or TXT", file_types=[".pdf", ".txt"])
        upload_btn = gr.Button("📥 Process File")

    upload_status = gr.Textbox(label="Upload & Indexing Status")

    with gr.Row():
        user_input = gr.Textbox(label="💬 Ask a question about the document")
        chat_btn = gr.Button("🔍 Get Answer")

    chat_output = gr.Textbox(label="🤖 Answer")

    upload_btn.click(process_file, inputs=file_input, outputs=upload_status)
    chat_btn.click(chat, inputs=user_input, outputs=chat_output)

# ✅ Launch the app with public link
demo.launch()
