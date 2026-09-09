from pathlib import Path
from sentence_transformers import SentenceTransformer

from pdf_loader import load_pdf
from chunker import chunk_pages


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    return model


def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    data_folder = Path(__file__).resolve().parent.parent / "data"

    pdf_files = list(data_folder.glob("*.pdf"))

    all_chunks = []

    # Load and chunk every PDF
    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)

        chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=75
        )

        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Load embedding model
    model = load_embedding_model()

    # Create vectors
    embeddings = create_embeddings(
        all_chunks,
        model
    )

    print("\nEmbedding generation complete!")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimensions per embedding: {len(embeddings[0])}")

    print("\nExample chunk:")
    print(all_chunks[0]["text"][:300])

    print("\nFirst 10 values of its embedding:")
    print(embeddings[0][:10])