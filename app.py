import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Bọc an toàn bộ ghi âm đa phương tiện
try:
    from st_audiorecorder import st_audiorecorder
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Cấu hình giao diện chuẩn cho một ứng dụng tra cứu luật chuyên nghiệp
st.set_page_config(page_title="Trợ Lý Luật AI", layout="wide", page_icon="⚖️")
st.title("⚖️ Hệ Thống Chatbot Tra Cứu Điều Khoản Luật Nội Bộ")

# Tự động sinh thư mục 'data' nếu chưa có để tránh lỗi đường dẫn
if not os.path.exists("data"):
    os.makedirs("data")

# --- QUẢN LÝ CẤU HÌNH API KEY TỰ ĐỘNG ---
st.sidebar.header("🔑 Khởi Tạo Hệ Thống")
google_api_key = ""

# Kiểm tra xem bạn đã cài cấu hình bảo mật trong mục Secrets chưa
if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Hệ thống đã kích hoạt bằng Key trong Secrets!")
else:
    # Nếu chưa cài Secrets mới hiện ô này ra trên giao diện để gõ tay
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# --- ĐIỀU KIỆN ĐỂ KÍCH HOẠT CHẠY ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng cấu hình hoặc nhập Google API Key ở thanh bên để kích hoạt trợ lý luật.", icon="🔑")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Hàm nạp dữ liệu văn bản từ thư mục 'data' (Sử dụng Cache và FAISS để chống lỗi hệ thống)
    @st.cache_resource
    def load_legal_documents():
        documents = []
        if os.path.exists("data"):
            # Tự động quét file luật định dạng PDF
            pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
            
            # Tự động quét file luật định dạng Word (.docx)
            docx_loader = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
            documents.extend(docx_loader.load())
            
            # Tự động quét file văn bản thuần (.txt)
            txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
        
        if not documents:
            return None

        # TỐI ƯU CHO LUẬT: Cắt
