import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Giao dien chinh - Viet tieu de ngan gon de khong bao gio bi loi be dong chuoi chu
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("AI Law Assistant - Chat va Ghi am tra cuu")

if not os.path.exists("data"):
    os.makedirs("data")

# --- 2. QUAN LY CAU HINH API KEY ---
st.sidebar.header("API Key Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- 3. DIEU KIEN KICH HOAT APP ---
if not google_api_key:
    st.info("Please enter Google Gemini API Key in the sidebar to start.")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Ham nap va phan tich tai lieu trong thu muc 'data'
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
        
        try:
            google_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=google_api_key
            )
            vectorstore = FAISS.from_documents(documents=splits, embedding=google_embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as embed_error:
            st.error(f"Embedding Error: {embed_error}")
            return None

    # DA SUA LOI CHU HOA TAI DAY: Doi tu 'St.spinner' thanh 'st.spinner' viet thuong chuan xac
    with st.spinner("Processing documents in 'data' folder..."):
        retriever = load_legal_documents()

    if retriever is None:
        st.warning("Thu muc 'data' dang trong! Hay de file vao muc 'data' tren GitHub.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else:
        st.sidebar.success("Database synced successfully!")
        if st.sidebar.button("Clear Cache"):
            st.
