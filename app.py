import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("🤖 AI Document Voice Assistant")

# Create data directory if it does not exist
if not os.path.exists("data"):
    os.makedirs("data")

# --- 2. SIDEBAR API KEY CONFIGURATION ---
st.sidebar.header("API Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- 3. MAIN APPLICATION LOGIC ---
if not google_api_key:
    st.info("Please configure or enter your Google Gemini API Key in the sidebar to activate the assistant.")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Document Processor function using FAISS Database
    @st.cache_resource
    def load_local_knowledge_base():
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
        # FIXED: Added the missing colon (:) at the end of the except line
        except Exception as embed_error:
            st.error(f"Embedding Error: {embed_error}")
            return None

    # Load and sync the documents inside data folder
    retriever = load_local_knowledge_base()

    if retriever is None:
        st.warning("The 'data' folder is empty! Please upload your documents (.pdf, .docx, .txt) into the 'data' folder on GitHub.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else
