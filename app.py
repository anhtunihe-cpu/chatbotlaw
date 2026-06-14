import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Cấu hình giao diện chatbot tra cứu luật
st.set_page_config(page_title="Trợ Lý Tra Cứu Luật", layout="wide", page_icon="⚖️")
st.title("⚖️ Hệ Thống Chatbot Tra Cứu Điều Khoản Luật Nội Bộ")

# Tự động tạo thư mục 'data' nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

# Menu bên trái cấu hình Key bảo mật
st.sidebar.header("🔑 Khởi Tạo Hệ Thống")
google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

if not google_api_key:
    st.info("Vui lòng nhập Google API Key ở thanh bên để kích hoạt trợ lý.", icon="🔑")
else:
    # Thiết lập biến môi trường cho API Key
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Hàm nạp tài liệu từ thư mục 'data' (Sử dụng Cache)
    @st.cache_resource
    def load_legal_documents():
        documents = []
        if os.path.exists("data"):
            # Quét file PDF luật
            pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
            
            # Quét file Word luật
            docx_loader = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
            documents.extend(docx_loader.load())
            
            # Quét file văn bản thuần TXT
            txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
        
        if not documents:
            return None

        # Cắt đoạn lớn (1500 ký tự) để giữ trọn vẹn ngữ cảnh Điều/Khoản luật
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        # Số hóa văn bản sử dụng mô hình Embedding của Google
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = Chroma.from_documents(documents=splits, embedding=google_embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    # Tiến hành nạp dữ liệu văn bản
    with st.spinner("Đang phân tích các văn bản luật trong thư mục 'data'..."):
        retriever = load_legal_documents()

    # Kiểm tra kho dữ liệu có trống
