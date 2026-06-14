import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. UI Setup
st.set_page_config(page_title="AI Law Assistant", layout="wide")
st.title("⚖️ AI Law Assistant")

# 2. Cấu hình API Key
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("Please enter your API Key in the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# 3. Document Processing (SỬA LỖI MODEL Ở ĐÂY)
@st.cache_resource
def load_legal_documents():
    if not os.path.exists("data"): return None
    
    documents = []
    for loader_cls in [PyPDFLoader, Docx2txtLoader, TextLoader]:
        loader = DirectoryLoader("data", glob=f"**/*.{'pdf' if loader_cls==PyPDFLoader else 'docx' if loader_cls==Docx2txtLoader else 'txt'}", loader_cls=loader_cls)
        documents.extend(loader.load())
    
    if not documents: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Sử dụng tên mô hình rút gọn 'text-embedding-004'
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

retriever = load_legal_documents()
...
