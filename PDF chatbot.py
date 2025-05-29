import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import requests
from io import BytesIO

# Load OpenAI key securely from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Please add your OpenAI API key in Streamlit secrets.")
    st.stop()

# --- Custom Page Config and Styling ---
st.set_page_config(page_title="Chat with Indian Constitution", page_icon="📜", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    body {background-color: #fff;}
    .main {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #0F52BA;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .ashoka {
        width: 80px;
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
        border-right: 4px solid #138808;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Ashoka_Chakra.svg/2048px-Ashoka_Chakra.svg.png", width=80)
st.markdown("<div class='title'>Chat with the Indian Constitution 🇮🇳</div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Section ---
with st.sidebar:
    st.subheader("📘 Document Summary")
    st.markdown("""
    This document is a curated version of the **Indian Constitution**.
    
    Use this app to ask questions and explore the content interactively.
    """)

    st.subheader("💡 Try asking:")
    st.markdown("""
    - What is Article 370 about?
    - When was the Indian Constitution adopted?
    - What are the Fundamental Rights?
    - Tell me about Directive Principles of State Policy.
    """)

    file = st.file_uploader("📎 Upload your Constitution PDF", type="pdf")

# --- PDF Handling ---
if file is not None:
    with st.spinner("⏳ Processing document..."):
        reader = PdfReader(file)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    if raw_text.strip() == "":
        st.error("⚠️ Could not extract text from the PDF.")
    else:
        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_text(raw_text)

        # Embedding & Vector DB
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # --- QA Interface ---
        st.subheader("💬 Ask a question:")
        user_question = st.text_input("Type your question here...")

        if user_question:
            with st.spinner("🤖 Generating answer..."):
                matches = vector_store.similarity_search(user_question)
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name="gpt-3.5-turbo"
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=matches, question=user_question)
                st.success("✅ Answer:")
                st.markdown(f"**{answer}**")
