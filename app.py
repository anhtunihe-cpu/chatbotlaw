import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from audiorecorder import audiorecorder  # Thư viện ghi âm chuẩn cho Streamlit Cloud
import openai
import os

st.set_page_config(page_title="AI Voice Đọc Tài Liệu", layout="wide", page_icon="🎙️")
st.title("🎙️ Trợ Lý AI Đọc Tài Liệu Bằng Giọng Nói (Gemini)")

# Cấu hình thanh bên trái (Sidebar)
st.sidebar.header("🔑 Cấu hình Khóa API")
google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")
openai_api_key = st.sidebar.text_input("Nhập OpenAI API Key (Chỉ dùng để dịch Giọng nói):", type="password")

if not google_api_key:
    st.info("Vui lòng điền Google Gemini API Key ở thanh bên trái để bắt đầu.", icon="🔑")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    if openai_api_key:
        openai.api_key = openai_api_key

    # Upload file trực tiếp từ giao diện web
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
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
            vectorstore = Chroma.from_documents(documents=splits, embedding=google_embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 4})

        with st.spinner("AI đang đọc hiểu tài liệu của bạn..."):
            retriever = process_uploaded_