import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Bọc an toàn thư viện ghi âm không cần OpenAI Key
try:
    from st_audiorecorder import st_audiorecorder
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

st.set_page_config(page_title="AI Voice Đọc Tài Liệu", layout="wide", page_icon="🎙️")
st.title("🎙️ Trợ Lý AI Đọc Tài Liệu Bằng Giọng Nói (Chỉ Dùng Gemini)")

# --- QUẢN LÝ CẤU HÌNH API KEY (CHỈ DÙNG GOOGLE) ---
st.sidebar.header("🔑 Cấu hình Khóa API")

if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Đã nhận diện Google API Key từ Secrets!")
else:
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# --- ĐIỀU KIỆN ĐỂ KÍCH HOẠT ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng điền hoặc cấu hình Google Gemini API Key ở thanh bên để bắt đầu.", icon="🔑")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

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
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                all_docs.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
            vectorstore = Chroma.from_documents(documents=splits, embedding=google_embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 4})

        with st.spinner("AI đang phân tích và đọc hiểu tài liệu của bạn..."):
            retriever = process_uploaded_files(uploaded_files)
            st.sidebar.success("Đã nạp kho dữ liệu tài liệu thành công!")

        # --- KHU VỰC GHI ÂM GIỌNG NÓI ---
        st.sidebar.write("---")
        st.sidebar.header("🎙️ Nói với Chatbot")
        
        user_query = ""  # Biến lưu trữ câu hỏi cuối cùng
        
        if HAS_AUDIO:
            audio = st_audiorecorder("Bấm để nói 🎤", "Đang nghe... Bấm lại để dừng 🛑")
            
            if len(audio) > 0:
                audio_file_path = "temp_audio.wav"
                audio.export(audio_file_path, format="wav")
                
                with st.spinner("Gemini đang nghe và dịch giọng nói của bạn..."):
                    try:
                        audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                        
                        with open(audio_file_path, "rb") as f:
