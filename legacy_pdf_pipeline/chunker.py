from pathlib import Path
from pdf_loader import load_pdf


def chunk_text(text, chunk_size=500, overlap=75):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_pages(pages, chunk_size=500, overlap=75):
    all_chunks = []

    for page in pages:
        text_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_number, chunk in enumerate(text_chunks, start=1):
            all_chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_number,
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    data_folder = Path(__file__).resolve().parent.parent / "data"

    pdf_files = list(data_folder.glob("*.pdf"))

    total_chunks = 0

    for pdf_file in pdf_files:
        print("=" * 60)
        print(f"Processing: {pdf_file.name}")

        pages = load_pdf(pdf_file)

        chunks = chunk_pages(pages)

        print(f"Pages: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

        total_chunks += len(chunks)

        if chunks:
            print("\nExample chunk:")
            print(f"Source: {chunks[0]['source']}")
            print(f"Page: {chunks[0]['page']}")
            print(f"Chunk ID: {chunks[0]['chunk_id']}")
            print()

            print(chunks[0]["text"][:700])

        print()

    print("=" * 60)
    print(f"Total chunks created: {total_chunks}")