# 🎬 YouTube Chatbot — RAG-Powered Video Q&A

A clean, minimal **Streamlit** web app that lets you ask questions about any YouTube video and get AI-generated answers using **Retrieval-Augmented Generation (RAG)**.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)

---

## 📌 What It Does

1. Takes a **YouTube Video ID** and a **language**
2. Extracts the video **transcript** using `YouTubeTranscriptApi`
3. Splits the transcript into **chunks** using LangChain
4. Generates **embeddings** and stores them in a **FAISS vector database**
5. Answers your questions using **RAG** — retrieves relevant chunks and generates answers via an LLM

---

## 🧠 Architecture

```
YouTube Video ID
      │
      ▼
┌─────────────────────┐
│ YouTubeTranscriptApi│  ← Fetch transcript
└────────┬────────────┘
         ▼
┌─────────────────────────────┐
│ RecursiveCharacterTextSplitter │  ← Split into chunks (1000 chars, 100 overlap)
└────────┬────────────────────┘
         ▼
┌─────────────────────┐
│   OpenAIEmbeddings  │  ← Generate vector embeddings
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   FAISS VectorStore │  ← Store & index embeddings
└────────┬────────────┘
         │
   User asks a question
         │
         ▼
┌─────────────────────┐
│  Retriever (k=6)    │  ← Find 6 most relevant chunks
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  ChatPromptTemplate │  ← Build prompt with context + question
│     +  ChatOpenAI   │  ← Generate answer via LLM
└─────────────────────┘
         │
         ▼
      Answer
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Transcript Extraction** | `youtube-transcript-api` |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **LLM** | OpenAI `gpt-4o-mini` (via OpenRouter) |
| **Orchestration** | LangChain (prompts, chains) |

---

## 📂 Project Structure

```
yt_chatbot/
├── app.py                  # Main Streamlit application
├── style.css               # Custom dark theme CSS
├── .env                    # API keys (not committed)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── .venv/                  # Virtual environment
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An API key from [OpenRouter](https://openrouter.ai/) (or OpenAI directly)

### 1. Clone the Repository

```bash
git clone https://github.com/imayank17/Youtube-chatbot.git
cd Youtube-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install streamlit faiss-cpu youtube-transcript-api langchain langchain-openai langchain-community langchain-text-splitters tiktoken python-dotenv
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🎯 How to Use

1. **Enter a Video ID** — Paste any YouTube video ID in the sidebar (e.g., `Gfr50f6ZBvo`)
2. **Select Language** — Choose the transcript language (English, Hindi, Spanish, etc.)
3. **Click "Process Video"** — This fetches the transcript and builds the vector store
4. **Ask a Question** — Type your question in the text area
5. **Click "Get Answer"** — The app retrieves relevant context and generates an answer

---

## 🔑 Key Implementation Details

### RAG Pipeline

The core of this project is the **Retrieval-Augmented Generation** pipeline:

- **Chunking Strategy**: `RecursiveCharacterTextSplitter` with 1000-character chunks and 100-character overlap ensures context is preserved across chunk boundaries.
- **Embedding Model**: `text-embedding-3-small` converts text chunks into 1536-dimensional vectors.
- **Vector Search**: FAISS performs efficient similarity search to find the top 6 most relevant chunks for any query.
- **Prompt Engineering**: A structured prompt template ensures the LLM answers based only on retrieved context.

### Session State Caching

The vector store is cached in `st.session_state` to avoid recomputing embeddings on every interaction — this makes follow-up questions instant.

### Error Handling

The app gracefully handles:
- Missing or disabled transcripts
- Invalid video IDs
- Missing API keys
- Network errors

---

## 🧪 Key Concepts Demonstrated

- **RAG (Retrieval-Augmented Generation)** — Grounding LLM responses in external data
- **Vector Databases** — Storing and searching high-dimensional embeddings
- **LangChain** — Building LLM-powered applications with modular components
- **Prompt Engineering** — Crafting effective prompts for accurate answers
- **Streamlit** — Rapid prototyping of ML-powered web apps
- **API Integration** — Working with YouTube, OpenAI, and OpenRouter APIs

---



## 🤝 Contributing

Feel free to fork this project, open issues, or submit pull requests. Contributions are welcome!

---

<p align="center">
  Built with ❤️ using Streamlit · LangChain · FAISS · OpenRouter
</p>
