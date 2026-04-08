import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Configuration

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
PROXY_URL = os.getenv("PROXY_URL")

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


def load_css():
    """Load external CSS file."""
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Core RAG Functions


def get_transcript(video_id: str, language: str = "en") -> str:
    """Fetch and concatenate the transcript of a YouTube video."""
    if PROXY_URL:
        ytt = YouTubeTranscriptApi(proxy_url=PROXY_URL)
    else:
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
        page_title="AskYoutube AI",
        page_icon="🎬",
        layout="centered",
    )

    load_css()

    # Title
    st.markdown('<h1 class="main-title">🎬 AskYoutube AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask questions about any YouTube video — powered by RAG</p>',
        unsafe_allow_html=True,
    )


    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠 Configuration")
        st.markdown("")

        video_id = st.text_input(
            "YouTube Video ID",
            placeholder="e.g. Gfr50f6ZBvo",
        )

        language_name = st.selectbox(
            "Transcript Language",
            options=list(LANGUAGES.keys()),
        )
        language_code = LANGUAGES[language_name]

        st.markdown("")
        process_btn = st.button("▶ Process Video", use_container_width=True)

        st.divider()

        if st.session_state.get("processed_video_id"):
            st.markdown(
                f'<span class="status-badge">● Video loaded</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"ID: `{st.session_state.processed_video_id}`")
        else:
            st.caption("💡 Process a video first, then ask questions in the main panel.")

    # Session State Initialization
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_video_id" not in st.session_state:
        st.session_state.processed_video_id = None

    # Process Video
    if process_btn:
        if not video_id.strip():
            st.warning("Please enter a valid YouTube Video ID.")
        elif not API_KEY:
            st.error("Missing `OPENROUTER_API_KEY` in your `.env` file.")
        else:
            with st.spinner("Fetching transcript & building vector store…"):
                try:
                    transcript = get_transcript(video_id.strip(), language_code)
                    vector_store = create_vectorstore(transcript)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_video_id = video_id.strip()
                    st.success(f"Video **{video_id.strip()}** processed in **{language_name}**!")
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video.")
                except NoTranscriptFound:
                    st.error(
                        f"No **{language_name}** transcript found. Try a different language."
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")




    # Question & Answer
    st.markdown('<p class="section-label">Your Question</p>', unsafe_allow_html=True)

    query = st.text_area(
        "Ask your question",
        placeholder="e.g. What are the key takeaways from this video?",
        label_visibility="collapsed",
        height=120,
    )

    answer_btn = st.button(
        "✦ Get Answer",
        disabled=(st.session_state.vector_store is None),
        use_container_width=True,
    )

    if answer_btn:
        if not query.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Generating answer…"):
                try:
                    answer = get_answer(query.strip(), st.session_state.vector_store)
                    st.markdown(
                        f'<div class="answer-card">'
                        f'<div class="answer-header">✦ Answer</div>'
                        f'<div class="answer-text">{answer}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # Footer
    st.markdown(
        '<div class="footer">Built with Streamlit · LangChain · FAISS · OpenRouter</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
