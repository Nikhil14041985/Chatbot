import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import requests
from io import BytesIO

# OpenAI API Key
OPENAI_API_KEY = "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"  # replace with your actual key

# Load PDF from GitHub (optional default)
GITHUB_PDF_URL = Constitution_India_subset.pdf

st.header("📄 Chat with your PDF")

# Sidebar: Upload PDF
with st.sidebar:
    st.title("Upload your document")
    file = st.file_uploader("Upload a PDF file", type="pdf")
    use_default = st.checkbox("Or use default PDF from GitHub")

# Use uploaded file or default from GitHub
if file is not None or use_default:
    st.info("Loading PDF...")
    
    # Load the file
    if file is not None:
        pdf_stream = file
    else:
        response = requests.get(GITHUB_PDF_URL)
        if response.status_code != 200:
            st.error("Failed to load PDF from GitHub.")
            st.stop()
        pdf_stream = BytesIO(response.content)

    # Extract text from PDF
    pdf_reader = PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text.strip():
        st.error("No extractable text found in the PDF.")
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings and FAISS index
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # User question input
        user_question = st.text_input("❓ Ask a question about the PDF:")

        if user_question:
            st.info("Searching and generating answer...")
            matches = vector_store.similarity_search(user_question)
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=matches, question=user_question)
            st.success("Answer:")
            st.write(response)
