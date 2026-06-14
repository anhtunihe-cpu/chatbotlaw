import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import base64

# Cấu hình giao diện chatbot
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
    st.sidebar.success("✅ Đã kích hoạt bằng Key bảo mật (Secrets)!")
else:
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# --- ĐIỀU KIỆN KÍCH HOẠT CHẠY ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng nhập hoặc cấu hình Google API Key ở thanh bên để kích hoạt trợ lý.", icon="🔑")
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

        # Cắt đoạn văn bản tối ưu ngữ cảnh luật
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = FAISS.from_documents(documents=splits, embedding=google_embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    with st.spinner("Hệ thống đang phân tích kho tài liệu văn bản luật trong mục 'data'..."):
        retriever = load_legal_documents()

    if retriever is None:
        st.warning("⚠️ Thư mục 'data' đang trống! Hãy copy file tài liệu luật (.pdf, .docx, .txt) vào mục 'data' rồi bấm nút bên dưới.")
        if st.sidebar.button("🔄 Tải lại dữ liệu"):
            st.rerun()
    else:
        st.sidebar.success("✅ Kho dữ liệu văn bản luật đã sẵn sàng!")
        if st.sidebar.button("🔄 Cập nhật tài liệu mới"):
            st.cache_resource.clear()
            st.rerun()

        # Quản lý và hiển thị lịch sử trò chuyện
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # --- KHU VỰC CHỨA TÍNH NĂNG GHI ÂM (HIỂN THỊ CỐ ĐỊNH) ---
        st.write("---")
        st.subheader("🎙️ Đ
