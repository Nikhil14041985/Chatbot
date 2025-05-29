import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load OpenAI key securely from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Please add your OpenAI API key in Streamlit secrets.")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="Chat with Indian Constitution", page_icon="📜", layout="wide")

# --- Custom Styling for Indian Constitution Theme ---
st.markdown("""
    <style>
        .main {
            background-color: #fffbe6;
            font-family: 'Georgia', serif;
            color: #222;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #1a4d2e;
        }
        .ashoka {
            width: 80px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 0.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
            border-left: 6px solid #138808;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header with Ashoka Chakra and Title ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Ashoka_Chakra.svg/2048px-Ashoka_Chakra.svg.png", width=80)
st.markdown("<div class='title'>Chat with the Indian Constitution 🇮🇳</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Source + Prompts
with st.sidebar:
    st.title("📘 Explore the Constitution")
    use_github = st.checkbox("Use PDF from GitHub", value=True)
    if not use_github:
        file = st.file_uploader("📂 Upload your PDF", type="pdf")

    # Prompts
    st.markdown("### 💡 Try Asking:")
    st.markdown("""
    - *What are the fundamental duties?*  
    - *What is the meaning of Article 21?*  
    - *Summarize the Preamble.*  
    - *Explain Directive Principles of State Policy.*
    """)

# Load file from GitHub or uploaded file
if use_github:
    GITHUB_PDF_URL = "https://raw.githubusercontent.com/Nikhil14041985/Chatbot/Main/Constitution_India_subset.pdf"
    response = requests.get(GITHUB_PDF_URL)
    file = BytesIO(response.content)

# --- Process the PDF ---
if file:
    with st.spinner("📖 Reading the PDF..."):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content

    if text.strip() == "":
        st.error("⚠️ No text found in the PDF.")
    else:
        # Text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)

        # --- Generate summary for sidebar ---
        summary = ""
        with st.spinner("🧾 Summarizing document..."):
            summary_input = " ".join(chunks[:3])
            summary_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
            summary_chain = load_qa_chain(summary_llm, chain_type="stuff")
            summary = summary_chain.run(input_documents=[], question=f"Summarize this in 3 lines: {summary_input}")

        # Sidebar Summary
        with st.sidebar:
            st.markdown("### 🧾 Brief Summary")
            st.info(summary)

        # --- Embedding and Search Setup ---
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # --- Main QA interface ---
        st.subheader("❓ Ask a Question from the Constitution")
        user_q = st.text_input("Type your question here:")

        if user_q:
            with st.spinner("🤖 Fetching answer..."):
                matches = vector_store.similarity_search(user_q)
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=matches, question=user_q)

                st.success("✅ Answer")
                st.markdown(f"**{answer}**")

