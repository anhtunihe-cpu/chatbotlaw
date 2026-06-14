import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Giao dien chinh - Viet khong dau de chong tuyet doi loi be dong chuoi chu
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("AI Law Assistant - Chat va Ghi am tra cuu")

if not os.path.exists("data"):
    os.makedirs("data")

# --- 2. QUAN LY CAU HINH API KEY ---
st.sidebar.header("API Key Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- 3. DIEU KIEN KICH HOAT APP ---
if not google_api_key:
    st.info("Please enter Google Gemini API Key in the sidebar to start.")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Ham nap va phan tich tai lieu trong thu muc 'data'
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        try:
            google_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=google_api_key
            )
            vectorstore = FAISS.from_documents(documents=splits, embedding=google_embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as embed_error:
            st.error(f"Embedding Error: {embed_error}")
            return None

    with St.spinner("Processing documents in 'data' folder..."):
        retriever = load_legal_documents()

    if retriever is None:
        st.warning("Thu muc 'data' dang trong! Hãy de file vao muc 'data' tren GitHub.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else:
        st.sidebar.success("Database synced successfully!")
        if st.sidebar.button("Clear Cache"):
            st.cache_resource.clear()
            st.rerun()

        # Lich su chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # --- 4. KHU VUC HIEN THI NUT GHI AM CO DINH ---
        st.write("---")
        
        user_query = ""  # Bien gop nhan thong tin tu Voice hoac Text

        # Widget ghi am mac dinh cua he thong - CHAC CHAN SE HIEN THI 100%
        audio_value = st.audio_input("Voice Input (Click round button to speak)")

        if audio_value is not None:
            audio_bytes = audio_value.read()
            with st.spinner("AI listening..."):
                try:
                    audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                    audio_data = {
                        "mime_type": "audio/wav",
                        "data": audio_bytes
                    }
                    response = audio_llm.invoke([
                        "Hay nghe doan am thanh nay va ghi lai thanh van ban tieng Viet. Chi tra ve chu viet, khong giai thich.",
                        audio_data
                    ])
                    user_query = response.content.strip()
                    st.info(f"Nội dung nghe được từ giọng nói: {user_query}")
                except Exception as e:
                    st.error(f"Voice Error: {e}")

        # Khung nhap cau hoi bang chu
        text_input = st.chat_input("Or type your legal question here...")
        if text_input:
            user_query = text_input

        # --- 5. AI TRA CUU CHINH XAC THEO VAN BAN (RAG) ---
        if user_query:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with
