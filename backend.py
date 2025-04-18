import streamlit as st
import fitz  # PyMuPDF for text extraction
import pdfplumber  # For table extraction
import faiss
import numpy as np
import pandas as pd
import os
import uuid  # Unique session ID
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from frontend import setup_ui

# Setup frontend UI
hf_api_key = setup_ui()

# Load Q&A file from CSV
def load_qa_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return {row["question"]: row["answer"] for _, row in df.iterrows()}
    except Exception as e:
        st.error(f"Error loading Q&A file: {e}")
        return {}

qa_data = load_qa_data("qa_data.csv")  # Load CSV-based Q&A data

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if not hf_api_key:
    st.warning("Please enter your Hugging Face API key to proceed.")
    st.stop()
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_table = page.extract_table()
            if extracted_table:
                tables.append("\n".join(["\t".join(str(cell) if cell else "") for row in extracted_table]))
    return tables

# Create FAISS index for embeddings
def create_faiss_index(text_data):
    text_chunks = text_data.split("\n\n")
    embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, text_chunks

# Retrieve relevant text from FAISS
def retrieve_relevant_text(query, index, text_chunks, top_k=3):
    query_embedding = np.array([embedding_model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0]]

# Load LLM
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.5})

# Define prompt for LLM
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

# Process uploaded PDF
if uploaded_file and hf_api_key:
    st.write("✅ PDF uploaded successfully!")
    pdf_path = f"uploaded_{st.session_state.session_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    extracted_text = extract_text_from_pdf(pdf_path)
    extracted_tables = extract_tables_from_pdf(pdf_path)
    all_text_data = extracted_text + "\n".join(extracted_tables)
    index, text_chunks = create_faiss_index(all_text_data)
    st.session_state.faiss_index = index
    st.session_state.text_chunks = text_chunks
    st.success("✅ PDF processed! You can now ask questions.")

# Answer user questions (First check CSV, then PDF, then LLM)
def get_answer(user_question):
    if user_question in qa_data:
        return qa_data[user_question]  # Return predefined answer from CSV
    elif st.session_state.faiss_index:  # Use PDF content
        relevant_chunks = retrieve_relevant_text(user_question, st.session_state.faiss_index, st.session_state.text_chunks)
        context = "\n".join(relevant_chunks)
        return qa_chain.run({"context": context, "question": user_question})
    else:
        return "I'm sorry, I couldn't find an answer."

# User input for Q&A
user_question = st.text_input("Ask a question:")
if user_question:
    answer = get_answer(user_question)
    st.session_state.chat_history.append({"question": user_question, "answer": answer})

    # Display chat history
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.write("---")
