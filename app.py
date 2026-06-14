import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Cấu hình giao diện chuẩn cho một ứng dụng tra cứu luật
st.set_page_config(page_title="Trợ Lý Ảo Tra Cứu Luật", layout="wide", page_icon="⚖️")
st.title("⚖️ Hệ Thống Chatbot Tra Cứu Điều Khoản Luật Nội Bộ")

# Tự động sinh thư mục 'data' nếu người dùng chưa tạo
if not os.path.exists("data"):
    os.makedirs("data")

# Menu bên trái để cấu hình thông tin Key bảo mật
st.sidebar.header("🔑 Khởi Tạo Hệ Thống")
google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key của bạn:", type="password")

if not google_api_key:
    st.info("Vui lòng nhập Google Gemini API Key ở thanh bên trái để kích hoạt trợ lý luật.", icon="🔑")
else:
    # Cấu hình API Key vào hệ thống
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Hàm tối ưu hóa (Cache) dùng để quét và nạp tất cả tài liệu trong thư mục data
    @st.cache_resource
    def load_legal_documents():
        documents = []
        
        # Tự động quét file PDF luật
        if os.path.exists("data"):
            pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
            
            # Tự động quét file Word luật (.docx)
            docx_loader = DirectoryLoader("data", glob="**/*.docx", loader_cls=Docx2txtLoader)
            documents.extend(docx_loader.load())
            
            # Tự động quét file văn bản thuần (.txt)
            txt_loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
        
        if not documents:
            return None

        # TỐI ƯU CHO LUẬT: Cắt đoạn lớn hơn (1500 ký tự) để giữ trọn vẹn ngữ cảnh một Điều Luật
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        # Số hóa và lưu vào cơ sở dữ liệu Chroma tạm thời
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = Chroma.from_documents(documents=splits, embedding=google_embeddings)
        
        # Trả về bộ rút trích thông tin liên quan nhất
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    # Tiến hành nạp dữ liệu văn bản từ thư mục data
    with st.spinner("Hệ thống đang quét và phân tích các văn bản luật trong mục 'data'..."):
        retriever = load_legal_documents()

    # Kiểm tra xem thư mục data đã có file luật nào chưa
    if retriever is None:
        st.warning("⚠️ Thư mục 'data' hiện đang trống! Hãy copy các file tài liệu luật (.pdf, .docx, .txt) vào thư mục 'data' nằm cùng cấp với file app.py này, sau đó bấm F5 để tải lại trang.")
    else:
        st.sidebar.success("✅ Đã đồng bộ kho
