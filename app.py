import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Bọc an toàn thư viện ghi âm chuẩn của Streamlit Cloud
try:
    from st_audiorecorder import st_audiorecorder
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Cấu hình giao diện chatbot tra cứu luật
st.set_page_config(page_title="Trợ Lý Tra Cứu Luật", layout="wide", page_icon="⚖️")
st.title("⚖️ Trợ Lý AI Tra Cứu Điều Khoản Luật (Hỗ Trợ Giọng Nói)")

# Tự động tạo thư mục 'data' nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

# --- QUẢN LÝ CẤU HÌNH API KEY ---
st.sidebar.header("🔑 Cấu hình Khóa API")
google_api_key = ""

# Ưu tiên kiểm tra Secrets trên Streamlit Cloud
if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Hệ thống đã kích hoạt bằng Key trong Secrets!")
else:
    google_api_key = st.sidebar.text_input("Nhập Google Gemini API Key:", type="password")

# --- ĐIỀU KIỆN KÍCH HOẠT CHẠY ỨNG DỤNG ---
if not google_api_key:
    st.info("Vui lòng nhập hoặc cấu hình Google API Key ở thanh bên để kích hoạt trợ lý.", icon="🔑")
else:
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
        
        google_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vectorstore = Chroma.from_documents(documents=splits, embedding=google_embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    # Tiến hành nạp dữ liệu văn bản
    with st.spinner("Đang phân tích văn bản luật trong thư mục 'data'..."):
        retriever = load_legal_documents()

    if retriever is None:
        st.warning("⚠️ Thư mục 'data' đang trống! Hãy copy file luật (.pdf, .docx, .txt) vào thư mục 'data' rồi bấm F5 tải lại.")
    else:
        st.sidebar.success("✅ Kho dữ liệu luật đã sẵn sàng!")
        
        if st.sidebar.button("🔄 Cập nhật lại tài liệu mới"):
            st.cache_resource.clear()
            st.rerun()

        # --- KHU VỰC NÚT BẤM GHI ÂM (VOICE INPUT) ---
        st.sidebar.write("---")
        st.sidebar.header("🎙️ Nói với Trợ Lý Luật")
        
        user_query = ""  # Biến gộp chung câu hỏi (Nói hoặc Gõ)
        
        if HAS_AUDIO:
            # Tạo nút ghi âm đẹp mắt ở thanh bên trái
            audio = st_audiorecorder("Bấm để nói 🎤", "Đang nghe... Bấm lại để dừng 🛑")
            
            if len(audio) > 0:
                audio_file_path = "temp_audio.wav"
                audio.export(audio_file_path, format="wav")
                
                with st.spinner("Gemini đang nghe và dịch giọng nói của bạn..."):
                    try:
                        audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                        with open(audio_file_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        audio_data = {"mime_type": "audio/wav", "data": audio_bytes}
                        response = audio_llm.invoke([
                            "Hãy nghe đoạn âm thanh này và dịch sang văn bản chữ viết tiếng Việt. Chỉ trả về văn bản, không giải thích gì thêm.",
                            audio_data
                        ])
                        user_query = response.content.strip()
                        st.sidebar.info(f"Nội dung nhận diện: *\"{user_query}\"*")
                    except Exception as e:
                        st.sidebar.error(f"Lỗi nhận diện giọng nói: {e}")
        else:
            st.sidebar.warning("⚠️ Thư viện âm thanh đang được tải ngầm từ máy chủ.")

        # --- QUẢN LÝ LỊCH SỬ CHAT ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Nhận câu hỏi từ ô gõ chữ (nếu không dùng giọng nói)
        text_input = st.chat_input("Nhập nội dung hoặc điều khoản luật cần tra cứu...")
        if text_input:
            user_query = text_input

        # --- XỬ LÝ AI TRẢ LỜI CHÍNH XÁC THEO LUẬT (RAG) ---
        if user_query:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("AI đang tra cứu văn bản luật để trích xuất..."):
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                        
                        system_prompt = """Bạn là một chuyên gia pháp lý chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các đoạn trích tài liệu luật dưới đây.

Yêu cầu nghiêm ngặt:
1. Trích dẫn rõ ràng tên văn bản, Chương, Điều, Khoản (nếu có thông tin) xuất hiện trong bối cảnh để làm căn cứ pháp lý.
2. Trả lời chi tiết, đầy đủ thông tin, không tóm tắt quá ngắn làm mất đi tính chính xác của điều khoản.
3. Nếu nội dung được hỏi không xuất hiện trong tài liệu, hãy trả lời chính xác là: 'Thông tin quy định này không có trong kho tài liệu luật bạn cung cấp'. Tuyệt đối không tự bịa ra câu trả lời.

Bối cảnh tài liệu luật trích xuất:
{context}"""
                        
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{question}"),
                        ])
                        
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        rag_chain = (
                            {"context": retriever | format_docs, "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        answer = rag_chain.invoke(user_query)
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
