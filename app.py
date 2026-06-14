import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
import os

# BỌC AN TOÀN: Thử import thư viện ghi âm, nếu lỗi thì bỏ qua không làm sập App
try:
    from st_audiorecorder import st_audiorecorder
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

st.set_page_config(page_title="AI Voice Đọc Tài Liệu", layout="wide", page_icon="🎙️")
st.title("🎙️ Trợ Lý AI Đọc Tài Liệu Thông Minh (Gemini)")

# --- QUẢN LÝ CẤU HÌNH API KEY ---
st.sidebar.header("🔑 Cấu hình Khóa API")

if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("✅ Đã nhận diện Google API Key từ Secrets!")
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

        # --- KHU VỰC GHI ÂM GIỌNG NÓI (VOICE INPUT) ---
        st.sidebar.write("---")
        st.sidebar.header("🎙️ Nói với Chatbot")
        
        user_query = ""  # Biến lưu trữ câu hỏi cuối cùng
        
        # Nếu hệ thống cài đặt thành công thư viện âm thanh thì mới hiển thị nút nói
        if HAS_AUDIO:
            if not openai_api_key:
                st.sidebar.warning("⚠️ Hãy nhập thêm OpenAI API Key để bật tính năng Nói.")
            else:
                audio = st_audiorecorder("Bấm để nói 🎤", "Đang nghe... Bấm lại để dừng 🛑")
                
                if len(audio) > 0:
                    audio_file_path = "temp_audio.mp3"
                    audio.export(audio_file_path, format="mp3")
                    
                    with st.spinner("AI đang lắng nghe và dịch giọng nói..."):
                        try:
                            with open(audio_file_path, "rb") as f:
                                transcript = openai.audio.transcriptions.create(
                                    model="whisper-1", 
                                    file=f,
                                    language="vi"
                                )
                            user_query = transcript.text
                            st.sidebar.info(f"Giọng nói nhận diện: *\"{user_query}\"*")
                        except Exception as e:
                            st.sidebar.error(f"Lỗi nhận diện giọng nói: {e}")
        else:
            st.sidebar.warning("⚠️ Máy chủ đang thiết lập micro phần cứng, tính năng nói tạm ẩn. Bạn hãy gõ chữ ở khung chat dưới màn hình nhé!")

        # --- QUẢN LÝ LỊCH SỬ CHAT ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        text_input = st.chat_input("Gõ nội dung cần hỏi vào đây...")
        if text_input:
            user_query = text_input

        # --- XỬ LÝ AI TRẢ LỜI TÀI LIỆU (RAG) ---
        if user_query:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Gemini đang truy vấn văn bản để trả lời..."):
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                        
                        system_prompt = (
                            "Bạn là một trợ lý ảo chuyên nghiệp, chuyên trả lời dựa trên tài liệu cung cấp.\n"
                            "Hãy sử dụng các đoạn bối cảnh dưới đây để trả lời câu hỏi.\n"
                            "Nếu thông tin không có trong tài liệu, hãy nói thật là 'Thông tin này không có trong tài liệu'.\n"
                            "Tuyệt đối không tự đoán hoặc bịa ra câu trả lời.\n\n"
                            "Bối cảnh tài liệu trích xuất:\n{context}"
                        )
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
