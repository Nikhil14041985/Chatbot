import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# ✅ Your OpenAI API Key (set this via secrets or env var in production)

st.secrets["openai"]["api_key"]

# ✅ Link to your GitHub PDF (RAW link)
GITHUB_PDF_URL = "https://raw.githubusercontent.com/Nikhil14041985/Chatbot/Main/Constitution_India_subset.pdf"

st.title("📄 Chat with Constitution of India")

# Download PDF from GitHub
response = requests.get(GITHUB_PDF_URL)
if response.status_code != 200:
    st.error("Failed to fetch PDF from GitHub.")
else:
    with st.spinner("Reading PDF from GitHub..."):
        pdf_reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Check if text is extractable
        if not text.strip():
            st.error("No text found in the PDF.")
        else:
            # Split the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = text_splitter.split_text(text)

            # Generate embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_texts(chunks, embeddings)

            # Input question
            user_question = st.text_input("❓ Ask a question:")

            if user_question:
                with st.spinner("Thinking..."):
                    matches = vector_store.similarity_search(user_question)
                    llm = ChatOpenAI(
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0,
                        model_name="gpt-3.5-turbo"
                    )
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer = chain.run(input_documents=matches, question=user_question)
                    st.success("Answer:")
                    st.write(answer)

