import time
import random

from google.genai import errors


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    context_parts = []


    for number, result in enumerate(
        results,
        start=1
    ):

        context = f"""
SOURCE {number}

Park:
{result['park']}

Topic:
{result['topic']}

Information:
{result['text']}

Original Source:
{result['source_title']}

Source Year:
{result['source_year']}

Source Type:
{result['source_type']}

Freshness Information:
{result['freshness_note']}
"""

        context_parts.append(
            context
        )


    return "\n".join(
        context_parts
    )


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    results,
    client
):

    context = build_context(
        results
    )


    prompt = f"""
You are ParkWise Bangladesh.

You are an AI assistant that answers questions about
Bangladesh national parks using a curated knowledge base.

You MUST answer using ONLY the information in the
DOCUMENT CONTEXT below.

RULES:

1. Do not use outside knowledge.

2. Never invent facts.

3. If the context does not contain enough information,
say exactly:

"I couldn't find enough information in the available
Bangladesh park dataset."

4. If information comes from an older management plan
or historical document, briefly mention that the
information may not represent current visitor conditions.

5. Do not invent opening hours, ticket prices,
transport schedules, current closures, phone numbers,
or current regulations.

6. Keep answers clear and easy to understand.

7. You may combine information from multiple retrieved
records when they support the question.

8. Do not claim that ParkWise is an official government
service.

USER QUESTION:

{question}


DOCUMENT CONTEXT:

{context}


ANSWER:
"""


    # =====================================================
    # GEMINI FALLBACK MODELS
    # =====================================================

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite"
    ]


    for model_name in models:

        try:

            response = (
                client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            )


            if response.text:

                return response.text


        except errors.ServerError as error:

            print(
                f"{model_name} unavailable:"
            )

            print(
                error
            )


            time.sleep(
                1
                + random.uniform(
                    0,
                    0.5
                )
            )


        except errors.ClientError as error:

            print(
                f"{model_name} client error:"
            )

            print(
                error
            )


        except Exception as error:

            print(
                f"Unexpected error "
                f"with {model_name}:"
            )

            print(
                error
            )


    # =====================================================
    # FALLBACK MESSAGE
    # =====================================================

    return (
        "ParkWise successfully retrieved relevant "
        "Bangladesh park information, but the AI "
        "generation service is temporarily unavailable."
    )