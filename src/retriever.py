from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_loader import load_pdf
from chunker import chunk_pages


MODEL_NAME = "all-MiniLM-L6-v2"


def load_all_chunks():
    data_folder = Path(__file__).resolve().parent.parent / "data"

    pdf_files = list(data_folder.glob("*.pdf"))

    all_chunks = []

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)

        chunks = chunk_pages(
            pages,
            chunk_size=250,
            overlap=50
        )

        all_chunks.extend(chunks)

    return all_chunks


def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def search(query, chunks, embeddings, model, top_k=5):
    import re

    query_lower = query.lower()

    # ---------------------------------
    # 1. Detect park
    # ---------------------------------
    park_filter = None

    if "redwood" in query_lower:
        park_filter = "redwood.pdf"

    elif "rainier" in query_lower:
        park_filter = "mount_rainier.pdf"

    elif "rocky mountain" in query_lower:
        park_filter = "rocky_mountain.pdf"


    # ---------------------------------
    # 2. Query expansion
    # ---------------------------------
    expanded_query = query

    if "pet" in query_lower or "pets" in query_lower or "dog" in query_lower:
        expanded_query += (
            " pets dogs allowed prohibited leash restrained trails"
        )


    # ---------------------------------
    # 3. Semantic embedding
    # ---------------------------------
    query_embedding = model.encode(
        expanded_query,
        normalize_embeddings=True
    )

    scores = np.dot(
        embeddings,
        query_embedding
    )


    # ---------------------------------
    # 4. Topic keywords
    # ---------------------------------
    pet_keywords = {
        "pet",
        "pets",
        "dog",
        "dogs",
        "leash",
        "restrained",
        "prohibited"
    }


    candidate_results = []


    for index, score in enumerate(scores):

        chunk = chunks[index]

        # Only search requested park
        if park_filter and chunk["source"] != park_filter:
            continue

        adjusted_score = float(score)

        # Convert chunk into real words
        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z]+\b",
                chunk["text"].lower()
            )
        )


        # ---------------------------------
        # 5. Keyword boost
        # ---------------------------------
        if (
            "pet" in query_lower
            or "pets" in query_lower
            or "dog" in query_lower
        ):

            hits = len(
                pet_keywords.intersection(chunk_words)
            )

            adjusted_score += hits * 0.12


        candidate_results.append({
            "score": adjusted_score,
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })


    # Highest score first
    candidate_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidate_results[:top_k]


if __name__ == "__main__":
    print("Loading ParkWise documents...")

    chunks = load_all_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings = build_embeddings(
        chunks,
        model
    )

    print("\nParkWise Semantic Search is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("Ask a question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = search(
            query,
            chunks,
            embeddings,
            model,
            top_k=3
        )

        print("\nTop Results:")
        print("=" * 70)

        for number, result in enumerate(results, start=1):

            print(f"\nResult {number}")

            print(f"Similarity: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Chunk: {result['chunk_id']}")

            print("\nText:")
            print(result["text"][:700])

            print("\n" + "-" * 70)

        print()