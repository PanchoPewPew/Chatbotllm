from transformers import pipeline
gpt_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Initialize Streamlit App
st.title("LLM Chatbot with RAG")
st.write("Chatbot powered by LangChain and a free GPT model.")

# Upload PDF files
uploaded_file = st.file_uploader("Upload a PDF file for knowledge base:", type="pdf")

if uploaded_file:
    # Load PDF content (Assume you already have a PDF parsing function)
    pdf_text = "Parsed content from uploaded PDF"  # Replace with actual PDF parsing logic

    # Create embeddings for RAG
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = Chroma.from_texts([pdf_text], embeddings)

    # Initialize Hugging Face pipeline for GPT
    gpt_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    llm = HuggingFacePipeline(pipeline=gpt_pipeline)

    # Set up LangChain RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever())

    # Chat Interface
    st.write("Ask a question!")
    user_query = st.text_input("Your Question:")

    if user_query:
        response = qa_chain.run(user_query)
        st.write("Response:", response)
