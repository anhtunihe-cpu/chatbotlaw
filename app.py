import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Law Assistant", layout="wide")
st.title("⚖️ AI Law Assistant")

# Cấu hình API Key từ Streamlit Secrets
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.info("Please enter your API Key in the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def load_legal_documents():
    if not os.path.exists("data"): return None
    
    documents = []
    # Quét file
    for loader_cls in [PyPDFLoader, Docx2txtLoader, TextLoader]:
        loader = DirectoryLoader("data", glob=f"**/*.{'pdf' if loader_cls==PyPDFLoader else 'docx' if loader_cls==Docx2txtLoader else 'txt'}", loader_cls=loader_cls)
        documents.extend(loader.load())
    
    if not documents: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # SỬA LỖI: Bỏ tiền tố 'models/' và dùng tên đơn giản hơn
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Embedding Error: {e}. Try changing the model name.")
        return None

retriever = load_legal_documents()

if not retriever:
    st.warning("No documents found or Embedding failed.")
else:
    # Voice Input
    audio_value = st.audio_input("Voice Input (Click to speak)")
    user_query = ""

    if audio_value:
        with st.spinner("Processing..."):
            audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            response = audio_llm.invoke([
                "Transcribe to Vietnamese.",
                {"mime_type": "audio/wav", "data": audio_value.read()}
            ])
            user_query = response.content.strip()
            st.info(f"Recognized: {user_query}")

    text_input = st.chat_input("Or type your question...")
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
