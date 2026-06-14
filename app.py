import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cấu hình giao diện
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("🤖 Chatbot Đọc Tài Liệu")

# 2. Cấu hình API Key (Lấy từ Secrets)
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Nhập Gemini API Key:", type="password")

if not api_key:
    st.info("Vui lòng nhập API Key trong sidebar để bắt đầu.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# 3. Pipeline RAG (Xử lý tài liệu)
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("data"): return None
    
    documents = []
    # Quét file trong thư mục data
    for cls in [PyPDFLoader, Docx2txtLoader, TextLoader]:
        loader = DirectoryLoader("data", glob=f"**/*.{'pdf' if cls==PyPDFLoader else 'docx' if cls==Docx2txtLoader else 'txt'}", loader_cls=cls)
        documents.extend(loader.load())
    
    if not documents: return None
    
    # Chia nhỏ dữ liệu
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    
    # Sử dụng mô hình ổn định nhất
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Lưu vào FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

retriever = load_knowledge_base()

# 4. Giao diện Chat
if not retriever:
    st.warning("Không tìm thấy tài liệu trong thư mục 'data/'. Hãy tải file lên GitHub.")
else:
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Hỏi về tài liệu của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            # Truy vấn và trả lời
            docs = retriever.invoke(prompt)
            context = "\n\n".join(d.page_content for d in docs)
            answer = llm.invoke(f"Context: {context}\n\nQuestion: {prompt}").content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
