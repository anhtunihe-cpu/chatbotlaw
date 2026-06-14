import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from st_audiorecorder import st_audiorecorder  # Thư viện ghi âm chuẩn tương thích Streamlit Cloud
import openai
import os

st.set_page_config(page_title="AI Voice Đọc Tài Liệu", layout="wide", page_icon="🎙️")
st.title("🎙️ Trợ Lý AI Đọc Tài Liệu Bằng Giọng Nói (Gemini)")

# --- QUẢN LÝ CẤU HÌNH API KEY ---
st.sidebar.header("🔑 Cấu hình Khóa API")

# Kiểm tra xem có cấu hình Secrets trên Streamlit Cloud không, nếu không thì bắt nhập thủ công
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Đã tự động nhận diện Google API Key từ Secrets!")
else:
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = openai_api_key
else:
    openai_api_key = st.sidebar.text_input("Nhập OpenAI API Key (Dịch giọng nói):", type="password")


# --- ĐIỀU KIỆN ĐỂ KÍCH HOẠT ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng điền hoặc cấu hình Google Gemini API Key để bắt đầu ứng dụng.", icon="🔑")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    if openai_api_key:
        openai.api_key = openai_api_key

    # --- TẢI FILE TÀI LIỆU TRÊN SIDEBAR ---
    uploaded_files = st.sidebar.file_uploader(
        "Tải lên tài liệu của bạn (Hỗ trợ PDF):", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        @st.cache_resource
        def process_uploaded_files(files):
            all_docs = []
            if not os.path.exists("temp_files"):
                os.makedirs("temp_files")
                
            for file in files:
                temp_path = os.path.join("temp_files", file.name)
                with open
