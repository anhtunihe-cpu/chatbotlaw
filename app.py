import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Cấu hình giao diện chatbot tra cứu luật
st.set_page_config(page_title="Trợ Lý Luật Giọng Nói", layout="wide", page_icon="⚖️")
st.title("⚖️ Trợ Lý AI Tra Cứu Luật Bằng Giọng Nói & Chữ")

# Tự động tạo thư mục 'data' nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

# --- QUẢN LÝ CẤU HÌNH API KEY TỰ ĐỘNG ---
st.sidebar.header("🔑 Khởi Tạo Hệ Thống")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Đã kích hoạt bằng Key trong Secrets!")
else:
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# --- ĐIỀU KIỆN KÍCH HOẠT CHẠY ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng cấu hình hoặc nhập Google API Key ở thanh bên để kích hoạt trợ lý luật.", icon="🔑")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Hàm nạp tài liệu từ thư mục 'data' sử dụng FAISS
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

        # Cắt đoạn văn bản tối ưu giữ ngữ cảnh luật
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = FAISS.
