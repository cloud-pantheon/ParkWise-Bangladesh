from pathlib import Path
import pymupdf


def load_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "source": Path(pdf_path).name,
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


if __name__ == "__main__":
    data_folder = Path(__file__).resolve().parent.parent / "data"

    pdf_files = list(data_folder.glob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for pdf_file in pdf_files:
        print("=" * 60)
        print(f"Reading: {pdf_file.name}")

        pages = load_pdf(pdf_file)

        print(f"Pages with text: {len(pages)}")

        if pages:
            print("\nFirst 500 characters:")
            print(pages[0]["text"][:500])

        print()