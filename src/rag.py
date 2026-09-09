import os
import time
import random
from google.genai import errors
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from retriever import load_all_chunks, build_embeddings, search


MODEL_NAME = "all-MiniLM-L6-v2"


def build_context(results):
    context_parts = []

    for number, result in enumerate(results, start=1):
        context = f"""
SOURCE {number}
Document: {result['source']}
Page: {result['page']}
Text:
{result['text']}
"""
        context_parts.append(context)

    return "\n".join(context_parts)


def generate_answer(question, results, client):
    context = build_context(results)

    prompt = f"""
You are ParkWise, a National Park information assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the documents do not contain enough information, say:
   "I couldn't find enough information in the available park documents."
4. Keep the answer clear and concise.
5. Do not invent sources or page numbers.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
"""

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite"
    ]

    for model_name in models:

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:
                return response.text

        except errors.ServerError as e:

            print(
                f"{model_name} temporarily unavailable: {e}"
            )

            # Small delay before switching models
            time.sleep(
                1 + random.uniform(0, 0.5)
            )

            continue

        except errors.ClientError as e:

            print(
                f"{model_name} client/quota error: {e}"
            )

            continue

        except Exception as e:

            print(
                f"Unexpected error using {model_name}: {e}"
            )

            continue


    return (
        "ParkWise successfully found relevant information "
        "in the park documents, but the AI generation service "
        "is currently unavailable."
    )


def print_sources(results):
    print("\nSources:")

    seen = set()

    for result in results:
        source_key = (
            result["source"],
            result["page"]
        )

        if source_key not in seen:
            print(
                f"- {result['source']} — Page {result['page']}"
            )

            seen.add(source_key)


if __name__ == "__main__":

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY was not found in .env")
        exit()

    print("Loading ParkWise...")

    # Gemini client
    client = genai.Client(api_key=api_key)

    # Load our PDF chunks
    chunks = load_all_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    # Load embedding model
    embedding_model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings for all chunks
    embeddings = build_embeddings(
        chunks,
        embedding_model
    )

    print("\nParkWise RAG is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        # RETRIEVAL
        results = search(
            question,
            chunks,
            embeddings,
            embedding_model,
            top_k=3
        )

        # GENERATION
        answer = generate_answer(
            question,
            results,
            client
        )

        print("\nParkWise:")
        print(answer)

        print_sources(results)

        print("\n" + "=" * 70 + "\n")