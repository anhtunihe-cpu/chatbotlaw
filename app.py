import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. UI Setup
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("⚖️ AI Law Assistant")

# 2. API Key Setup - Cấu trúc lại để nhận diện Key ổn định nhất
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("Please enter your API Key in the sidebar.")
    st.stop()

# Đẩy thẳng vào biến môi trường hệ thống
os.environ["GOOGLE_API_KEY"] = api_key

# 3. Document Processing với cơ chế bẫy lỗi
@st.cache_resource
def load_legal_documents():
    if not os.path.exists("data"): return None
    
    documents = []
    # Quét dữ liệu
    for loader_cls in [PyPDFLoader, Docx2txtLoader, TextLoader]:
        loader = DirectoryLoader("data", glob=f"**/*.{'pdf' if loader_cls==PyPDFLoader else 'docx' if loader_cls==Docx2txtLoader else 'txt'}", loader_cls=loader_cls)
        documents.extend(loader.load())
    
    if not documents: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Sử dụng mô hình mới nhất và truyền trực tiếp key
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

retriever = load_legal_documents()

# 4. App Flow & Chat Interface
if not retriever:
    st.warning("No documents found in 'data/' folder or Embedding failed.")
else:
    # Voice Input - Đặt cố định
    user_query = ""
    audio_value = st.audio_input("Voice Input (Click to speak)")

    if audio_value:
        with st.spinner("Processing voice..."):
            audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            response = audio_llm.invoke([
                "Transcribe this audio to Vietnamese text. Return only the text.",
                {"mime_type": "audio/wav", "data": audio_value.read()}
            ])
            user_query = response.content.strip()
            st.info(f"Recognized: {user_query}")

    text_input = st.chat_input("Or type your legal question...")
    if text_input: user_query = text_input

    if user_query:
        if "messages" not in st.session_state: st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"): st.write(user_query)
        
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=api_key)
            prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
            
            chain = (
                {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            answer = chain.invoke(user_query)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
