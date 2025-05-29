import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key securely
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Please add your OpenAI API key in Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("📄 Chat with a PDF")

# Use GitHub PDF or Upload
with st.sidebar:
    st.title("Select Document")
    use_github = st.checkbox("Use PDF from GitHub", value=True)
    if not use_github:
        file = st.file_uploader("Upload your PDF", type="pdf")

# Load PDF from GitHub if selected
if use_github:
    GITHUB_PDF_URL = "https://raw.githubusercontent.com/Nikhil14041985/Chatbot/Main/Constitution_India_subset.pdf"
    response = requests.get(GITHUB_PDF_URL)
    file = BytesIO(response.content)

if file:
    with st.spinner("📖 Reading the PDF..."):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content

    if text.strip() == "":
        st.error("No text found in PDF.")
    else:
        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Ask user
        user_q = st.text_input("❓ Ask a question from the PDF")

        if user_q:
            with st.spinner("🤖 Getting answer..."):
                matches = vector_store.similarity_search(user_q)
                llm = ChatOpenAI(
                    temperature=0,
                    model="gpt-3.5-turbo",
                    api_key=OPENAI_API_KEY
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                answer = chain.run(input_documents=matches, question=user_q)
                st.success("Answer:")
                st.write(answer)
