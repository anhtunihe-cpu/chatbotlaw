# 📚 Chatbot RAG Đọc Tài Liệu Thông Minh

Ứng dụng Streamlit cho phép chat với tài liệu nội bộ (PDF, DOCX, TXT) sử dụng Google Gemini + FAISS.

## Kiến trúc hệ thống

```
data/ (tài liệu)
  │
  ▼
[Document Loader] → PDF / DOCX / TXT
  │
  ▼
[Text Splitter] → Chunk 800 ký tự, overlap 150
  │
  ▼
[Gemini Embedding] → models/text-embedding-004
  │
  ▼
[FAISS Vector Store] (in-memory)
  │
  ├──── Truy vấn người dùng (Text / Voice)
  │           │
  │           ▼
  │     [Similarity Search] → Top-4 chunks
  │           │
  ▼           ▼
[Gemini 1.5 Flash] ← Prompt + Context + History
  │
  ▼
[Câu trả lời + Trích dẫn nguồn]
```

## Cài đặt & Chạy

### 1. Clone & Cài thư viện
```bash
git clone <repo-url>
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Cấu hình API Key (không hardcode)
```bash
# Tạo file secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Điền API key thực vào secrets.toml
```

Hoặc dùng biến môi trường:
```bash
export GOOGLE_API_KEY="your-key-here"
```

### 3. Thêm tài liệu
```bash
# Copy file vào thư mục data/
cp your_document.pdf data/
cp your_document.docx data/
cp your_document.txt data/
```

### 4. Chạy app
```bash
streamlit run app.py
```

## Tính năng

| Tính năng | Chi tiết |
|-----------|---------|
| 📄 Tải tài liệu | Tự động quét toàn bộ `data/` |
| 🔍 Tìm kiếm | FAISS similarity search, top-4 chunks |
| 🤖 AI Model | Gemini 1.5 Flash với fallback tự động |
| 📎 Trích dẫn | Hiển thị tên file + số trang nguồn |
| 💬 Chat lịch sử | Giữ 6 lượt hội thoại gần nhất |
| 🎙️ Giọng nói | Google Speech Recognition (vi-VN) |
| 🔄 Reboot | Clear cache và tải lại tài liệu |
| 🛡️ Error handling | Xử lý ValidationError, 404, quota |

## Xử lý lỗi thường gặp

### Lỗi 404 – Model not found
- Hệ thống tự động thử các model theo thứ tự: `gemini-1.5-flash` → `gemini-1.5-flash-latest` → `gemini-pro`
- Kiểm tra API key có quyền truy cập Gemini API không

### Lỗi ValidationError (Pydantic)
- Kiểm tra phiên bản: `pip show langchain-google-genai`
- Khuyến nghị: `langchain-google-genai>=1.0.6`
- Đảm bảo tham số `model` dùng prefix `models/` với embedding

### Thêm tài liệu mới
1. Copy file vào `data/`
2. Nhấn nút **"Tải lại tài liệu (Reboot)"** trên sidebar
3. Hệ thống sẽ tự động index lại toàn bộ

## Cấu trúc dự án

```
rag-chatbot/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .gitignore
├── README.md
├── .streamlit/
│   ├── secrets.toml          # API keys (KHÔNG commit)
│   └── secrets.toml.example  # Template
└── data/                     # Thư mục tài liệu
    ├── noi_quy_cong_ty.txt   # Ví dụ
    ├── hop_dong_lao_dong.pdf
    └── ...
```
