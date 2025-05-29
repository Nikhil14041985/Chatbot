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

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("📄 Know Indian Constitution")

# Sidebar: File source + summary placeholder
with st.sidebar:
    st.title("📘 PDF Summary & Help")
    use_github = st.checkbox("Indian Constitution", value=True)
    if not use_github:
        file = st.file_uploader("Upload your PDF", type="pdf")

    # Suggested prompts (static)
    st.markdown("### 💡 Example Prompts")
    st.markdown("""
    - *What are the key rights mentioned?*  
    - *Summarize the duties of citizens.*  
    - *What is the structure of government outlined?*
    - *Explain the preamble in simple terms.*
    """)

# Load file from GitHub or upload
if use_github:
    GITHUB_PDF_URL = "https://raw.githubusercontent.com/Nikhil14041985/Chatbot/Main/Constitution_India_subset.pdf"
    response = requests.get(GITHUB_PDF_URL)
    file = BytesIO(response.content)

# Process PDF
if file:
    with st.spinner("📖 Reading the PDF..."):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content

    if text.strip() == "":
        st.error("No text found in the PDF.")
    else:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_text(text)

        # Sidebar Summary: Use first few chunks to summarize
        summary = ""
        with st.spinner("✍️ Generating brief summary..."):
            summary_input = " ".join(chunks[:3])
            summary_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
            summary_chain = load_qa_chain(summary_llm, chain_type="stuff")
            summary = summary_chain.run(input_documents=[], question=f"Summarize this in 3 lines: {summary_input}")

        # Show summary in sidebar
        with st.sidebar:
            st.markdown("### 🧾 Brief Summary")
            st.info(summary)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Main UI: Accept user question
        user_q = st.text_input("❓ Ask a question from the PDF")

        if user_q:
            with st.spinner("🤖 Thinking..."):
                matches = vector_store.similarity_search(user_q)
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=matches, question=user_q)
                st.success("Answer:")
                st.write(answer)
