

import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

#configuration

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
}

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the 
provided context from a YouTube video transcript. Use the context to give 
a detailed, accurate answer. If the context doesn't contain relevant 
information, say so.

Context:
{context}

Question: {question}

Answer:"""
)

CUSTOM_CSS = """
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }

    .subtitle {
        text-align: center;
        color: #8b95a5;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #c9d1d9;
    }

    /* Answer card */
    .answer-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .answer-card h4 {
        color: #58a6ff;
        margin-bottom: 0.5rem;
    }

    .answer-card p {
        color: #c9d1d9;
        line-height: 1.7;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #484f58;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #21262d;
    }
</style>
"""


# Core RAG Functions


def get_transcript(video_id: str, language: str = "en") -> str:
    """Fetch and concatenate the transcript of a YouTube video."""
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=[language])
    transcript_data = fetched.to_raw_data()
    return " ".join(chunk["text"] for chunk in transcript_data)


def create_vectorstore(transcript: str):
    """Split transcript into chunks and store embeddings in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="openai/text-embedding-3-small",
    )
    return FAISS.from_documents(chunks, embeddings)


def get_answer(query: str, vector_store) -> str:
    """Retrieve relevant context and generate an answer using the LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="openai/gpt-4o-mini",
    )
    chain = RAG_PROMPT | llm
    result = chain.invoke({"context": context, "question": query})
    return result.content


# Streamlit UI


def main():
    st.set_page_config(
        page_title="YouTube Chatbot",
        page_icon="🎥",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="main-title">🎥 YouTube Chatbot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask questions about any YouTube video — powered by RAG</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("")

        video_id = st.text_input(
            "🔗 YouTube Video ID",
            placeholder="e.g. Gfr50f6ZBvo",
        )

        language_name = st.selectbox(
            "🌐 Transcript Language",
            options=list(LANGUAGES.keys()),
        )
        language_code = LANGUAGES[language_name]

        st.markdown("")
        process_btn = st.button("🔄 Process Video", use_container_width=True)

        st.divider()

        if st.session_state.get("processed_video_id"):
            st.markdown(
                f'<span class="status-badge">✅ Video loaded</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"ID: `{st.session_state.processed_video_id}`")
        else:
            st.caption("ℹ️ Process a video first, then ask questions in the main panel.")

    # Session State Initialization
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_video_id" not in st.session_state:
        st.session_state.processed_video_id = None

    # Process Video
    if process_btn:
        if not video_id.strip():
            st.warning("⚠️ Please enter a valid YouTube Video ID.")
        elif not API_KEY:
            st.error("🔑 Missing OPENROUTER_API_KEY in your .env file.")
        else:
            with st.spinner("📥 Fetching transcript & building vector store…"):
                try:
                    transcript = get_transcript(video_id.strip(), language_code)
                    vector_store = create_vectorstore(transcript)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_video_id = video_id.strip()
                    st.success(f"✅ Video **{video_id.strip()}** processed in **{language_name}**!")
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except NoTranscriptFound:
                    st.error(
                        f"❌ No **{language_name}** transcript found. Try a different language."
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Status
    if st.session_state.vector_store is not None:
        st.info(f"📌 Ready to answer questions about video: **{st.session_state.processed_video_id}**")
    else:
        st.info("👈 Enter a Video ID in the sidebar and click **Process Video** to get started.")

    st.divider()

    # Question & Answer
    st.subheader("💬 Ask a Question")

    query = st.text_area(
        "Ask your question",
        placeholder="What is the main topic of this video?",
        label_visibility="collapsed",
        height=100,
    )

    answer_btn = st.button(
        "🚀 Get Answer",
        disabled=(st.session_state.vector_store is None),
        use_container_width=True,
    )

    if answer_btn:
        if not query.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    answer = get_answer(query.strip(), st.session_state.vector_store)
                    st.markdown(
                        f'<div class="answer-card">'
                        f"<h4>📝 Answer</h4>"
                        f"<p>{answer}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")

    # Footer
    st.markdown(
        '<div class="footer">Built with Streamlit • LangChain • FAISS • OpenRouter</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
