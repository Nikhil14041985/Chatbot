import streamlit as st
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load API key securely from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    st.error("Please add your OpenAI API key in Streamlit secrets.")
    st.stop()

st.header("📄 Chat with a GitHub PDF")

# Example hosted PDF (raw GitHub URL)
GITHUB_PDF_URL = "https://raw.githubusercontent.com/Nikhil14041985/Chatbot/Main/Constitution_India_subset.pdf"

with st.sidebar:
    st.title("Document Source")
    use_github_pdf = st.checkbox("Use hosted GitHub PDF", value=True)

# Load PDF from GitHub or allow user upload
if use_github_pdf:
    with st.spinner("Downloading PDF from GitHub..."):
        response = requests.get(GITHUB_PDF_URL)
        file = BytesIO(response.content)
else:
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    if text.strip() == "":
        st.error("No extractable text found in the PDF.")
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Ask question
        user_question = st.text_input("❓ Ask a question about the document:")

        if user_question:
            with st.spinner("Thinking..."):
                matches = vector_store.similarity_search(user_question)
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name="gpt-3.5-turbo",
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=matches, question=user_question)
                st.success("Answer:")
                st.write(response)

