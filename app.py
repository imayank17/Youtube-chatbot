

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
    # Retrieve top-k relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt and invoke the LLM
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

    # ── Title ──
    st.markdown(
        "<h1 style='text-align:center;'>🎥 YouTube Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:gray;'>"
        "Ask questions about any YouTube video — powered by RAG"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        video_id = st.text_input(
            "Enter YouTube Video ID",
            placeholder="e.g. Gfr50f6ZBvo",
        )
        language = st.selectbox(
            "Select Transcript Language",
            options=["en", "hi", "es", "fr"],
        )
        process_btn = st.button("🔄 Process Video", use_container_width=True)

        st.divider()
        st.caption("ℹ️ Process a video first, then ask questions in the main panel.")

    # ── Session State Initialization ──
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_video_id" not in st.session_state:
        st.session_state.processed_video_id = None

    # ── Process Video ──
    if process_btn:
        if not video_id.strip():
            st.warning("⚠️ Please enter a valid YouTube Video ID.")
        elif not API_KEY:
            st.error("🔑 Missing OPENROUTER_API_KEY in your .env file.")
        else:
            with st.spinner("📥 Fetching transcript & building vector store…"):
                try:
                    transcript = get_transcript(video_id.strip(), language)
                    vector_store = create_vectorstore(transcript)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_video_id = video_id.strip()
                    st.success(f"✅ Video **{video_id.strip()}** processed successfully!")
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except NoTranscriptFound:
                    st.error(
                        f"❌ No transcript found for language **'{language}'**. "
                        "Try a different language."
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Show current status ──
    if st.session_state.vector_store is not None:
        st.info(f"📌 Ready to answer questions about video: **{st.session_state.processed_video_id}**")
    else:
        st.info("👈 Enter a Video ID in the sidebar and click **Process Video** to get started.")

    st.divider()

    # ── Question & Answer ──
    st.subheader("💬 Ask a Question")

    query = st.text_area(
        "Ask your question",
        placeholder="What is the main topic of this video?",
        label_visibility="collapsed",
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
                    st.markdown("---")
                    st.markdown("### 📝 Answer")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")


if __name__ == "__main__":
    main()
