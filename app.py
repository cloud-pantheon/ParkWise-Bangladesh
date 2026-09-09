import os
import sys
import streamlit as st

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# =========================================================
# IMPORT SRC MODULES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(
    os.path.join(BASE_DIR, "src")
)

from retriever import (
    MODEL_NAME,
    load_all_records,
    build_embeddings,
    search
)

from rag import generate_answer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ParkWise Bangladesh",
    page_icon="🇧🇩",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 14px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 13px;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Streamlit Cloud support
try:
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

except Exception:
    pass


if not api_key:

    st.error(
        "Gemini API key was not found. "
        "Add GEMINI_API_KEY to your .env file "
        "or Streamlit deployment secrets."
    )

    st.stop()


client = genai.Client(
    api_key=api_key
)


# =========================================================
# INITIALIZE RAG
# =========================================================

@st.cache_resource
def initialize_rag():

    records = load_all_records()

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = build_embeddings(
        records,
        embedding_model
    )

    return (
        records,
        embedding_model,
        embeddings
    )


with st.spinner(
    "Loading Bangladesh Parks knowledge base..."
):

    (
        records,
        embedding_model,
        embeddings
    ) = initialize_rag()


# =========================================================
# CHAT SESSION
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🇧🇩 ParkWise")

    st.caption(
        "Bangladesh National Parks RAG Assistant"
    )

    st.divider()


    # -----------------------------------------------------
    # PARKS
    # -----------------------------------------------------

    st.subheader("🌿 Current Parks")

    st.write(
        "🌳 Lawachara National Park"
    )

    st.write(
        "🌲 Satchari National Park"
    )

    st.write(
        "🌴 Bhawal National Park"
    )


    st.divider()


    # -----------------------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------------------

    st.subheader(
        "📚 Knowledge Base"
    )

    st.metric(
        "Knowledge Records",
        len(records)
    )

    st.caption(
        "Embedding model"
    )

    st.code(
        MODEL_NAME,
        language=None
    )


    st.divider()


    # -----------------------------------------------------
    # RETRIEVAL EVALUATION
    # -----------------------------------------------------

    st.subheader(
        "🧪 Retrieval Evaluation"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Top-1",
            "100%"
        )

    with col2:

        st.metric(
            "Top-3",
            "100%"
        )

    st.caption(
        "18-question curated benchmark"
    )


    st.divider()


    # -----------------------------------------------------
    # DEBUG
    # -----------------------------------------------------

    debug_mode = st.toggle(
        "🔧 Developer Debug Mode",
        value=False
    )


    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()

    st.caption(
        "Educational project. "
        "Not an official Bangladesh Forest "
        "Department application."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🇧🇩🌿 ParkWise Bangladesh
    </div>

    <div class="subtitle">
        Explore Bangladesh's national parks through
        a Retrieval-Augmented Generation system.
    </div>
    """,
    unsafe_allow_html=True
)


st.info(
    "ParkWise searches a curated knowledge base derived "
    "from Bangladesh Forest Department and BFIS materials "
    "before generating its answer."
)


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        "#### 💡 Try asking"
    )

    examples = [
        "What animals live in Bhawal National Park?",
        "Where is Lawachara National Park?",
        "What visitor facilities are available at Satchari?",
        "What type of forest is Bhawal National Park?"
    ]

    for example in examples:

        st.markdown(
            f"- {example}"
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # -------------------------------------------------
        # DISPLAY STORED SOURCES
        # -------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 View Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
### 🌿 {source["park"]}

**Topic:** `{source["topic"]}`

**Source:** {source["source_title"]}

**Year:** {
    source["source_year"]
    if source["source_year"]
    else "Not specified"
}

**Relevance:** {source["score"]:.2f}
"""
                    )

                    if source.get(
                        "freshness_note"
                    ):

                        st.caption(
                            source["freshness_note"]
                        )

                    if source.get(
                        "source_url"
                    ):

                        st.markdown(
                            f"[🔗 Open original source]"
                            f"({source['source_url']})"
                        )

                    st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask about wildlife, forests, locations, facilities..."
)


if question:


    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # =====================================================
    # RETRIEVAL
    # =====================================================

    results = search(
        question,
        records,
        embeddings,
        embedding_model,
        top_k=5
    )


    # =====================================================
    # DEBUG MODE
    # =====================================================

    if debug_mode:

        with st.expander(
            "🔍 Retrieval Debug",
            expanded=False
        ):

            for rank, result in enumerate(
                results,
                start=1
            ):

                st.markdown(
                    f"""
### Result {rank}

**Park:** {result["park"]}

**Topic:** `{result["topic"]}`

**Score:** `{result["score"]:.3f}`

**Record ID:** `{result["id"]}`
"""
                )

                st.write(
                    result["text"]
                )

                st.caption(
                    f"Source: "
                    f"{result['source_title']}"
                )

                st.divider()


    # =====================================================
    # GENERATION
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching Bangladesh park information..."
        ):

            answer = generate_answer(
                question,
                results,
                client
            )


        st.markdown(
            answer
        )


        # =================================================
        # SOURCE FILTERING
        # =================================================

        selected_sources = []

        if results:

            best_score = results[0]["score"]


            # Keep sources reasonably close
            # to strongest retrieval.
            relevant_results = [

                result

                for result in results

                if result["score"]
                >= best_score - 0.20
            ]


            # Maximum 3 source cards
            relevant_results = (
                relevant_results[:3]
            )


            # Remove duplicate records
            seen = set()


            for result in relevant_results:

                key = result["id"]

                if key in seen:
                    continue

                seen.add(key)


                selected_sources.append({

                    "id":
                        result["id"],

                    "park":
                        result["park"],

                    "topic":
                        result["topic"],

                    "score":
                        result["score"],

                    "source_title":
                        result["source_title"],

                    "source_url":
                        result["source_url"],

                    "source_year":
                        result["source_year"],

                    "freshness_note":
                        result["freshness_note"]
                })


        # =================================================
        # DISPLAY SOURCES
        # =================================================

        if selected_sources:

            with st.expander(
                "📚 View Sources"
            ):

                for source in selected_sources:

                    st.markdown(
                        f"""
### 🌿 {source["park"]}

**Topic:** `{source["topic"]}`

**Source:** {source["source_title"]}

**Year:** {
    source["source_year"]
    if source["source_year"]
    else "Not specified"
}

**Relevance:** {source["score"]:.2f}
"""
                    )


                    if source.get(
                        "freshness_note"
                    ):

                        st.caption(
                            source[
                                "freshness_note"
                            ]
                        )


                    if source.get(
                        "source_url"
                    ):

                        st.markdown(
                            f"[🔗 Open original source]"
                            f"({source['source_url']})"
                        )


                    st.divider()


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.messages.append({

        "role":
            "assistant",

        "content":
            answer,

        "sources":
            selected_sources
    })