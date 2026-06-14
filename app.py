import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Cau hinh giao dien chatbot
st.set_page_config(page_title="AI Chatbot Law", layout="wide")
st.title("AI Assistant - Tra cuu van ban luat")

# Tu dong tao thu muc 'data' neu chua co
if not os.path.exists("data"):
    os.makedirs("data")

# --- QUAN LY CAU HINH API KEY (UU TIEN XU LY TRUOC) ---
st.sidebar.header("API Key Configuration")
google_api_key = ""

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"] != "":
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets!")
else:
    google_api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- DIEU KIEN KICH HOAT CHAY APP ---
if not google_api_key:
    st.info("Please enter Google Gemini API Key in the sidebar to start.")
else:
    # PHAI NAP BIEN MOI TRUONG TRUOC KHI KHAI BAO HAM LUU TRU CACHE
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Ham nap tai lieu dung FAISS va Cache
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

        # Cat nho van ban giu ngu canh dieu khoan luat
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)
        
        try:
            # So hoa van ban su dung mo hinh Embeddings chinh xac
            google_embeddings = GoogleGenerativeAIEmbeddings(
                model="text-embedding-004", 
                google_api_key=google_api_key
            )
            vectorstore = FAISS.from_documents(documents=splits, embedding=google_embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as embed_error:
            st.error(f"Embedding Service Error: {embed_error}")
            return None

    # Tien hanh quet thu muc nap du lieu van ban luat
    with st.spinner("Processing documents in 'data' folder..."):
        retriever = load_legal_documents()

    # Kiem tra trang thai co so du lieu
    if retriever is None:
        st.warning("Database is empty or API authentication failed. Please upload files to 'data' folder and double check your API Key.")
        if st.sidebar.button("Reload Data"):
            st.rerun()
    else:
        st.sidebar.success("Database synced successfully!")
        if st.sidebar.button("Clear Cache"):
            st.cache_resource.clear()
            st.rerun()

        # Quan ly va hien thi lich su chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # --- KHU VUC GHI AM GIONG NOI ---
        st.write("---")
        user_query = "" 

        # Cong cu thu am phan cung mac dinh cua Streamlit
        audio_value = st.audio_input("Voice Input (Click to speak)")

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
                        "Hay nghe doan am thanh nay va ghi lai thanh van ban chu viet tieng Viet chinh xac nhat. Chi tra ve van ban chu viet, khong giai thich gi them.",
                        audio_data
                    ])
                    user_query = response.content.strip()
                    st.info(f"Nội dung nhận diện được: {user_query}")
                except Exception as e:
                    st.error(f"Voice Error: {e}")

        # Khung nhap chu thông thuong
        text_input = st.chat_input("Or type your legal question here...")
        if text_input:
            user_query = text_input

        # --- AI XU LY TRA CUU TAI LIEU (RAG) ---
        if user_query:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("AI searching database..."):
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=google_api_key)
                        
                        system_prompt = """Bạn là một chuyên gia pháp lý chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên tài liệu được cung cấp dưới đây.

Yêu cầu:
1. Trích dẫn cụ thể tên tài liệu, Điều, Khoản xuất hiện trong bối cảnh văn bản để làm căn cứ pháp lý rõ ràng.
2. Trả lời chi tiết, đầy đủ, không tóm tắt quá ngắn làm mất đi tính chính xác của điều khoản luật.
3. Nếu thông tin không có trong tài liệu, hãy trả lời chính xác là: 'Thông tin này không có trong tài liệu bạn cung cấp'. Không tự suy đoán hoặc bịa ra câu trả lời.

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
