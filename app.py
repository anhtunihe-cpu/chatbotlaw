import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Giao diện chính
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("⚖️ AI Law Assistant")

# 2. Cấu hình API Key an toàn
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("Please enter your API Key in the sidebar.")
    st.stop()

# 3. Xử lý tài liệu (RAG Pipeline) - Đã sửa lỗi Pydantic
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("data"): return None
    
    documents = []
    for cls in [PyPDFLoader, Docx2txtLoader, TextLoader]:
        loader = DirectoryLoader("data", glob=f"**/*.{'pdf' if cls==PyPDFLoader else 'docx' if cls==Docx2txtLoader else 'txt'}", loader_cls=cls)
        documents.extend(loader.load())
    
    if not documents: return None
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    
    # SỬA LỖI: Chỉ định rõ model và truyền trực tiếp API Key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

retriever = load_knowledge_base()

# 4. Chat Interface
if not retriever:
    st.warning("No documents found in 'data/' folder. Please upload files.")
else:
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Hỏi về tài liệu của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            context = "\n\n".join(d.page_content for d in retriever.invoke(prompt))
            answer = llm.invoke(f"Dựa trên tài liệu này: {context}\n\nHãy trả lời câu hỏi: {prompt}").content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
