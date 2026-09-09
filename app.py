import os
import sys
import streamlit as st

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# IMPORT FILES FROM SRC
# ---------------------------------------------------------

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from retriever import (
    load_all_chunks,
    build_embeddings,
    search
)

from rag import generate_answer


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="ParkWise AI",
    page_icon="🏞️",
    layout="centered"
)


# ---------------------------------------------------------
# SOURCE DISPLAY NAMES
# ---------------------------------------------------------

SOURCE_NAMES = {
    "redwood.pdf":
        "Redwood National & State Parks",

    "mount_rainier.pdf":
        "Mount Rainier National Park",

    "rocky_mountain.pdf":
        "Rocky Mountain National Park"
}


def pretty_source(filename):
    return SOURCE_NAMES.get(
        filename,
        filename
    )


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key was not found. "
        "Check your .env file."
    )
    st.stop()


client = genai.Client(
    api_key=api_key
)


# ---------------------------------------------------------
# LOAD RAG SYSTEM
# ---------------------------------------------------------

@st.cache_resource
def initialize_rag():

    chunks = load_all_chunks()

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = build_embeddings(
        chunks,
        embedding_model
    )

    return (
        chunks,
        embedding_model,
        embeddings
    )


with st.spinner(
    "Loading ParkWise knowledge base..."
):

    (
        chunks,
        embedding_model,
        embeddings
    ) = initialize_rag()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🏞️ ParkWise AI")

st.markdown(
    """
    Ask questions about U.S. National Parks using
    information retrieved from official National Park
    Service documents.
    """
)

st.info(
    "ParkWise answers using its document knowledge base "
    "instead of relying only on the language model."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🏞️ ParkWise")

    st.write(
        "A beginner-friendly Retrieval-Augmented "
        "Generation project."
    )

    st.divider()

    st.subheader("Available Parks")

    st.write("🌲 Redwood National & State Parks")
    st.write("🌋 Mount Rainier National Park")
    st.write("🏔️ Rocky Mountain National Park")

    st.divider()

    st.subheader("Knowledge Base")

    st.metric(
        "Document chunks",
        len(chunks)
    )

    st.caption(
        "Embedding model: all-MiniLM-L6-v2"
    )

    st.divider()

    # Developer debug switch
    debug_mode = st.toggle(
        "🔧 Developer Debug Mode"
    )

    st.divider()

    # Clear Chat
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 View Sources"
            ):

                for source in message["sources"]:

                    park_name = pretty_source(
                        source["source"]
                    )

                    st.markdown(
                        f"""
**🌿 {park_name}**

Page **{source['page']}**

Relevance: **{source['score']:.2f}**
"""
                    )

                    st.divider()


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask about pets, hiking, wildlife, camping, safety..."
)


if question:

    # ---------------------------------------------
    # USER MESSAGE
    # ---------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # ---------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------

    results = search(
        question,
        chunks,
        embeddings,
        embedding_model,
        top_k=5
    )


    # ---------------------------------------------
    # DEBUG PANEL
    # ---------------------------------------------

    if debug_mode:

        with st.expander(
            "🔍 Retrieval Debug",
            expanded=False
        ):

            for result in results:

                st.markdown(
                    f"""
**{pretty_source(result['source'])}**

Page: `{result['page']}`

Chunk: `{result['chunk_id']}`

Score: `{result['score']:.3f}`
"""
                )

                st.write(
                    result["text"]
                )

                st.divider()


    # ---------------------------------------------
    # GENERATE ANSWER
    # ---------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching park documents..."
        ):

            answer = generate_answer(
                question,
                results,
                client
            )

        st.markdown(answer)


        # -----------------------------------------
        # FILTER SOURCES
        # -----------------------------------------

        source_results = []

        if results:

            best_score = results[0]["score"]

            # Only display sources reasonably close
            # to the strongest retrieved result.
            for result in results:

                if (
                    result["score"]
                    >= best_score - 0.18
                ):

                    source_results.append(
                        result
                    )


        # -----------------------------------------
        # REMOVE DUPLICATE PAGE SOURCES
        # -----------------------------------------

        unique_sources = []

        seen = set()

        for result in source_results:

            key = (
                result["source"],
                result["page"]
            )

            if key not in seen:

                unique_sources.append({
                    "source":
                        result["source"],

                    "page":
                        result["page"],

                    "score":
                        result["score"]
                })

                seen.add(key)


        # -----------------------------------------
        # DISPLAY SOURCES
        # -----------------------------------------

        if unique_sources:

            with st.expander(
                "📚 View Sources"
            ):

                for source in unique_sources:

                    park_name = pretty_source(
                        source["source"]
                    )

                    st.markdown(
                        f"""
**🌿 {park_name}**

📄 Page **{source['page']}**

🎯 Relevance **{source['score']:.2f}**
"""
                    )

                    st.divider()


    # ---------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # ---------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": unique_sources
    })