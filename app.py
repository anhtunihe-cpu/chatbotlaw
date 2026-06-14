import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. UI Configuration
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("AI Law Assistant - Chat and Voice Search")

if not os.path.exists("data"):
    os.makedirs("data")

# 2. API Key Configuration
st.sidebar.header("API Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# 3. Main Application Logic
if not google_api_key:
    st.info("Please enter Google Gemini API Key in the sidebar to start.")
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key

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

    retriever = load_legal_documents()

    if retriever is None:
        st.warning("The 'data' folder is empty. Upload documents there.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else:
        st.sidebar.success("Database synced successfully!")
        
        # Audio Input and Text Input
        st.write("---")
        user_query = "" 
        audio_value = st.audio_input("Voice Input (Click the red circle to speak)")

        if audio_value is not None:
            audio_bytes = audio_value.read()
            try:
                audio_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                response = audio_llm.invoke([
                    "Listen to this audio and transcribe to Vietnamese. Return only the text.",
                    {"mime_type": "audio/wav", "data": audio_bytes}
                ])
                user_query = response.content.strip()
                st.info(f"Speech Recognized: {user_query}")
            except Exception as e:
                st.error(f"Voice Error: {e}")

        text_input = st.chat_input("Or type your question here...")
        if text_input:
            user_query = text_input

        # Chat Logic
        if user_query:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=google_api_key)
                
                system_prompt = """You are a legal expert assistant. Answer accurately based on the provided documents.
                Cite specific sections, Articles, and Titles. If not found, say: 'Thông tin này không có trong tài liệu'.
                Document Context: {context}"""
                
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                
                rag_chain = (
                    {"context": (retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))), "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                
                answer = rag_chain.invoke(user_query)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
