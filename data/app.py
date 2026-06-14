"""
Chatbot RAG Đọc Tài Liệu Thông Minh
=====================================
Kiến trúc: LangChain + FAISS + Google Gemini
Tác giả: Generated per BRD requirements
"""

import os
import glob
import streamlit as st
from pathlib import Path

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Chatbot Tài Liệu Thông Minh",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# IMPORTS (lazy để bắt lỗi thiếu thư viện sớm)
# ──────────────────────────────────────────────
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
    )
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_api_key() -> str | None:
    """Lấy API key từ Streamlit Secrets hoặc biến môi trường (không hardcode)."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GOOGLE_API_KEY")


def load_documents(data_dir: str = "data") -> list:
    """
    Load tất cả tài liệu từ thư mục data/.
    Hỗ trợ: .pdf, .docx, .txt
    """
    documents = []
    loaders_map = {
        "**/*.pdf":  PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.txt":  TextLoader,
    }

    data_path = Path(data_dir)
    if not data_path.exists():
        st.warning(f"⚠️ Thư mục `{data_dir}/` không tồn tại. Vui lòng tạo thư mục và thêm tài liệu.")
        return []

    for pattern, LoaderClass in loaders_map.items():
        for file_path in data_path.glob(pattern):
            try:
                if LoaderClass == TextLoader:
                    loader = LoaderClass(str(file_path), encoding="utf-8")
                else:
                    loader = LoaderClass(str(file_path))
                docs = loader.load()
                # Gắn metadata nguồn tài liệu
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                documents.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ Không thể đọc file `{file_path.name}`: {e}")

    return documents


def format_docs_with_sources(docs) -> str:
    """Format các đoạn văn bản kèm nguồn để đưa vào prompt."""
    result = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Không rõ nguồn")
        page = doc.metadata.get("page", "")
        page_info = f" (Trang {page + 1})" if page != "" else ""
        result.append(f"[Đoạn {i} | Nguồn: {source}{page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(result)


# ──────────────────────────────────────────────
# CACHED RESOURCES
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Đang xây dựng FAISS vector store...")
def build_vector_store(api_key: str):
    """
    Pipeline Ingestion:
    Load files → Split → Embed → FAISS
    Cache_resource: chỉ chạy 1 lần, tái sử dụng cho toàn bộ phiên.
    """
    # 1. Load documents
    documents = load_documents("data")
    if not documents:
        return None, 0

    # 2. Text splitting (chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # 3. Embedding – dùng text-embedding-004 (model mới nhất, tránh lỗi 404)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # Khai báo đúng prefix "models/"
            google_api_key=api_key,
        )
    except Exception as e:
        # Fallback sang embedding-001 nếu model mới không khả dụng
        st.warning(f"⚠️ Fallback embedding model: {e}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )

    # 4. Build FAISS in-memory vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


@st.cache_resource(show_spinner="🤖 Đang khởi tạo mô hình AI...")
def get_llm(api_key: str):
    """
    Khởi tạo Gemini 1.5 Flash LLM.
    Có xử lý lỗi 404 nếu model không tồn tại trên tài khoản.
    """
    model_candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-pro",
    ]
    for model_name in model_candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            # Test ping nhẹ
            llm.invoke("Hi")
            return llm, model_name
        except Exception as e:
            err_str = str(e).lower()
            if "404" in err_str or "not found" in err_str or "not_found" in err_str:
                continue  # Thử model tiếp theo
            else:
                raise e  # Lỗi khác thì throw ngay

    raise RuntimeError("Không tìm thấy model Gemini khả dụng. Kiểm tra API key hoặc quota.")


# ──────────────────────────────────────────────
# RAG CHAIN
# ──────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh, chuyên trả lời câu hỏi dựa CHÍNH XÁC vào nội dung tài liệu nội bộ được cung cấp.

**Nguyên tắc bắt buộc:**
1. CHỈ trả lời dựa trên các đoạn tài liệu trong phần [CONTEXT] bên dưới.
2. PHẢI trích dẫn nguồn tài liệu và điều khoản cụ thể trong câu trả lời (ví dụ: "Theo Điều 4 của Nội quy Công ty ABC...").
3. Nếu thông tin KHÔNG CÓ trong tài liệu, hãy trả lời thẳng thắn: "Tôi không tìm thấy thông tin này trong tài liệu hiện có."
4. KHÔNG được tự bịa đặt hoặc suy luận ngoài phạm vi tài liệu.
5. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.

[CONTEXT]
{context}
[/CONTEXT]

Lịch sử hội thoại:
{chat_history}
"""

def build_rag_chain(vector_store, llm):
    """
    Xây dựng RAG pipeline dùng RunnablePassthrough (LangChain LCEL).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # Lấy top 4 đoạn liên quan nhất
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_history(messages: list) -> str:
        if not messages:
            return "Chưa có lịch sử hội thoại."
        history_str = []
        for msg in messages[-6:]:  # Chỉ giữ 6 lượt gần nhất để tránh quá dài
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_str.append(f"{role}: {msg['content']}")
        return "\n".join(history_str)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_with_sources(
                retriever.invoke(x["question"])
            ),
            chat_history=lambda x: format_history(x.get("history", [])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ──────────────────────────────────────────────
# UI STYLES
# ──────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        /* Main container */
        .main { background: #0f1117; }

        /* Chat messages */
        .user-bubble {
            background: linear-gradient(135deg, #1e3a5f, #2563eb);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        }
        .bot-bubble {
            background: #1e2130;
            color: #e2e8f0;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 85%;
            border: 1px solid #2d3748;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .source-badge {
            display: inline-block;
            background: #7c3aed22;
            color: #a78bfa;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            border: 1px solid #7c3aed44;
            margin: 4px 2px 0;
        }
        .status-bar {
            background: #1e2130;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 12px;
            font-size: 0.85rem;
        }
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

def render_sidebar(api_key_ok: bool, n_chunks: int, model_name: str):
    with st.sidebar:
        st.markdown("## 📚 Chatbot Tài Liệu RAG")
        st.markdown("---")

        # Status
        col1, col2 = st.columns(2)
        with col1:
            if api_key_ok:
                st.success("🔑 API Key ✓")
            else:
                st.error("🔑 API Key ✗")
        with col2:
            if n_chunks > 0:
                st.success(f"📄 {n_chunks} chunks")
            else:
                st.warning("📄 Chưa có data")

        st.markdown(f"**Model:** `{model_name or 'N/A'}`")
        st.markdown("---")

        # Reboot / Clear cache
        st.markdown("### ⚙️ Quản lý hệ thống")
        if st.button("🔄 Tải lại tài liệu (Reboot)", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📁 Tài liệu đang load")

        data_path = Path("data")
        if data_path.exists():
            files = list(data_path.glob("**/*"))
            files = [f for f in files if f.is_file()]
            if files:
                for f in files:
                    ext = f.suffix.upper()
                    icons = {".PDF": "📕", ".DOCX": "📘", ".TXT": "📄"}
                    icon = icons.get(ext, "📎")
                    st.markdown(f"{icon} `{f.name}`")
            else:
                st.info("Chưa có file nào trong `data/`")
        else:
            st.warning("Thư mục `data/` chưa tồn tại")

        st.markdown("---")
        st.markdown("""
        <small>💡 <b>Hướng dẫn:</b><br>
        • Thêm file vào thư mục <code>data/</code><br>
        • Nhấn "Tải lại tài liệu" để cập nhật<br>
        • Hỗ trợ: PDF, DOCX, TXT</small>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    inject_css()

    # ── Kiểm tra dependencies ──
    if not DEPS_OK:
        st.error(f"❌ Thiếu thư viện: `{IMPORT_ERROR}`")
        st.code("pip install -r requirements.txt", language="bash")
        st.stop()

    # ── Lấy API Key ──
    api_key = get_api_key()
    if not api_key:
        st.error("❌ Không tìm thấy `GOOGLE_API_KEY`.")
        st.markdown("""
        **Cách cấu hình (không hardcode):**

        **Option 1 – Streamlit Secrets** (khuyến nghị cho production):
        Tạo file `.streamlit/secrets.toml`:
        ```toml
        GOOGLE_API_KEY = "your-api-key-here"
        ```

        **Option 2 – Biến môi trường** (cho development):
        ```bash
        export GOOGLE_API_KEY="your-api-key-here"
        streamlit run app.py
        ```
        """)
        st.stop()

    # ── Khởi tạo session state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Build vector store & LLM ──
    with st.spinner(""):
        try:
            vector_store, n_chunks = build_vector_store(api_key)
            llm, model_name = get_llm(api_key)
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo hệ thống: {e}")
            if st.button("🔄 Thử lại"):
                st.cache_resource.clear()
                st.rerun()
            st.stop()

    # ── Sidebar ──
    render_sidebar(bool(api_key), n_chunks, model_name)

    # ── Build RAG chain ──
    if vector_store is None:
        st.warning("⚠️ Chưa có tài liệu nào. Thêm file vào thư mục `data/` và nhấn **Tải lại tài liệu**.")
        rag_chain = None
        retriever = None
    else:
        rag_chain, retriever = build_rag_chain(vector_store, llm)

    # ── Header ──
    st.markdown("# 📚 Chatbot Đọc Tài Liệu Thông Minh")
    st.markdown("*Hỏi bất cứ điều gì về tài liệu nội bộ – AI sẽ trích dẫn nguồn chính xác.*")

    if n_chunks > 0:
        st.markdown(
            f'<div class="status-bar">✅ Đã index <b>{n_chunks} đoạn văn bản</b> | '
            f'Model: <code>{model_name}</code> | '
            f'Vector DB: <code>FAISS in-memory</code></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Hiển thị lịch sử chat ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    st.markdown("**📎 Nguồn tham khảo:**")
                    seen = set()
                    for doc in msg["sources"]:
                        src = doc.metadata.get("source_file", "Không rõ")
                        page = doc.metadata.get("page", "")
                        label = f"{src}" + (f" – Trang {page+1}" if page != "" else "")
                        if label not in seen:
                            seen.add(label)
                            st.markdown(f'<span class="source-badge">📄 {label}</span>', unsafe_allow_html=True)

    # ── Input tabs: Text & Voice ──
    tab_text, tab_voice = st.tabs(["💬 Nhập văn bản", "🎙️ Nhập giọng nói"])

    user_question = None

    with tab_text:
        user_question = st.chat_input("Nhập câu hỏi của bạn về tài liệu...")

    with tab_voice:
        st.info("🎙️ Tính năng nhập giọng nói yêu cầu chạy app trên máy local với microphone.")
        audio_col1, audio_col2 = st.columns([1, 3])
        with audio_col1:
            use_voice = st.button("🔴 Bắt đầu ghi âm", use_container_width=True)
        with audio_col2:
            st.markdown("*(Nhấn nút, nói câu hỏi, nhấn lại để dừng)*")

        if use_voice:
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("🎙️ Đang lắng nghe... Hãy nói câu hỏi của bạn.")
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source, timeout=10, phrase_time_limit=30)
                try:
                    voice_text = r.recognize_google(audio, language="vi-VN")
                    st.success(f"✅ Đã nhận dạng: **{voice_text}**")
                    user_question = voice_text
                except sr.UnknownValueError:
                    st.error("❌ Không thể nhận dạng giọng nói. Vui lòng thử lại.")
                except sr.RequestError as e:
                    st.error(f"❌ Lỗi dịch vụ nhận dạng: {e}")
            except ImportError:
                st.error("❌ Chưa cài `SpeechRecognition`. Chạy: `pip install SpeechRecognition`")
            except OSError:
                st.error("❌ Không tìm thấy microphone. Kiểm tra thiết bị âm thanh.")

    # ── Xử lý câu hỏi ──
    if user_question and user_question.strip():
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            if rag_chain is None:
                answer = "⚠️ Chưa có tài liệu nào trong hệ thống. Vui lòng thêm file vào thư mục `data/` và tải lại."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
            else:
                with st.spinner("🔍 Đang tìm kiếm trong tài liệu..."):
                    try:
                        # Retrieve sources để hiển thị
                        source_docs = retriever.invoke(user_question)

                        # Run RAG chain
                        response = rag_chain.invoke({
                            "question": user_question,
                            "history": st.session_state.messages[:-1],  # Bỏ câu hỏi vừa thêm
                        })

                        st.markdown(response)

                        # Hiển thị nguồn
                        if source_docs:
                            st.markdown("**📎 Nguồn tham khảo:**")
                            seen = set()
                            for doc in source_docs:
                                src = doc.metadata.get("source_file", "Không rõ")
                                page = doc.metadata.get("page", "")
                                label = f"{src}" + (f" – Trang {page+1}" if page != "" else "")
                                if label not in seen:
                                    seen.add(label)
                                    st.markdown(f'<span class="source-badge">📄 {label}</span>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": source_docs,
                        })

                    except Exception as e:
                        err_msg = str(e)
                        # Xử lý các loại lỗi phổ biến
                        if "ValidationError" in err_msg:
                            error_response = f"⚠️ Lỗi cấu hình model (ValidationError): {err_msg}\n\nVui lòng nhấn **Tải lại tài liệu** để thử lại."
                        elif "404" in err_msg or "NOT_FOUND" in err_msg:
                            error_response = "⚠️ Model AI không khả dụng (404). Hệ thống đang thử chuyển sang model dự phòng. Vui lòng nhấn **Tải lại tài liệu**."
                            st.cache_resource.clear()
                        elif "quota" in err_msg.lower() or "rate" in err_msg.lower():
                            error_response = "⚠️ Đã vượt giới hạn API quota. Vui lòng thử lại sau vài giây."
                        else:
                            error_response = f"❌ Lỗi không xác định: {err_msg}"

                        st.error(error_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_response,
                            "sources": [],
                        })

        st.rerun()


if __name__ == "__main__":
    main()
