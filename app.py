import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Cau hinh giao dien chatbot
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("AI Assistant - Tra cuu van ban luat")

# Tu dong tao thu muc 'data' neu chua co
if not os.path.exists("data"):
    os.makedirs("data")

# --- QUAN LY CAU HINH API KEY ---
st.sidebar.header("API Key Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- DIEU KIEN KICH HOAT APP ---
if not google_api_key:
    st.info("Please enter Google Gemini API Key in the sidebar to start.")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Ham nap tai lieu dung FAISS va Cache
    @st.cache_resource
    def load_legal_documents():
        documents = []
        if os.path.exists("data"):
            pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
            
            docx_loader = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
            documents.extend(docx_loader.load())
            
            txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
        
        if not documents:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = FAISS.from_documents(documents=splits, embedding=google_embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    with st.spinner("Processing documents in 'data' folder..."):
        retriever = load_legal_documents()

    if retriever is None:
        st.warning("Data folder is empty! Please upload .pdf or .docx files into 'data' folder on GitHub.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else:
        st.sidebar.success("Database synced successfully!")
        if st.sidebar.button("Clear Cache"):
            st.cache_resource.clear
