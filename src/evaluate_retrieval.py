import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retriever import (
    MODEL_NAME,
    load_all_records,
    build_embeddings,
    search
)


# ---------------------------------------------------------
# LOAD EVALUATION DATASET
# ---------------------------------------------------------

def load_evaluation_questions():

    eval_path = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "data"
        / "bd_parks_eval_questions.jsonl"
    )

    questions = []

    with open(
        eval_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                questions.append(
                    json.loads(line)
                )

    return questions


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Loading ParkWise Bangladesh..."
    )

    # Load knowledge records
    records = load_all_records()

    print(
        f"Knowledge records: {len(records)}"
    )


    # Load embedding model
    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )


    # Build embeddings
    print(
        "Creating embeddings..."
    )

    embeddings = build_embeddings(
        records,
        model
    )


    # Load evaluation questions
    eval_questions = (
        load_evaluation_questions()
    )

    print(
        f"\nRunning "
        f"{len(eval_questions)} "
        f"evaluation questions...\n"
    )


    # Counters
    top1_correct = 0
    top3_correct = 0

    failed_questions = []


    # -----------------------------------------------------
    # LOOP THROUGH QUESTIONS
    # -----------------------------------------------------

    for number, item in enumerate(
        eval_questions,
        start=1
    ):

        question = item["question"]

        expected_park = (
            item["target_park"]
        )

        expected_topics = set(
            item["expected_topics"]
        )


        results = search(
            question,
            records,
            embeddings,
            model,
            top_k=3
        )


        # Safety
        if not results:

            print(
                f"\nQuestion {number}: "
                f"{question}"
            )

            print(
                "❌ No results retrieved."
            )

            failed_questions.append(
                item
            )

            continue


        # -------------------------------------------------
        # TOP 1
        # -------------------------------------------------

        first = results[0]

        top1_match = (
            first["park"]
            == expected_park
            and
            first["topic"]
            in expected_topics
        )


        if top1_match:

            top1_correct += 1


        # -------------------------------------------------
        # TOP 3
        # -------------------------------------------------

        top3_match = any(

            result["park"]
            == expected_park

            and

            result["topic"]
            in expected_topics

            for result in results
        )


        if top3_match:

            top3_correct += 1


        # Save failed Top-1 questions
        if not top1_match:

            failed_questions.append({
                "question":
                    question,

                "expected_park":
                    expected_park,

                "expected_topics":
                    list(expected_topics),

                "retrieved_park":
                    first["park"],

                "retrieved_topic":
                    first["topic"],

                "score":
                    first["score"]
            })


        # -------------------------------------------------
        # PRINT QUESTION RESULT
        # -------------------------------------------------

        print(
            "=" * 70
        )

        print(
            f"Question {number}: "
            f"{question}"
        )

        print()

        print(
            f"Expected Park: "
            f"{expected_park}"
        )

        print(
            "Expected Topic(s): "
            + ", ".join(
                expected_topics
            )
        )

        print()

        print(
            "Top Result:"
        )

        print(
            f"Park: "
            f"{first['park']}"
        )

        print(
            f"Topic: "
            f"{first['topic']}"
        )

        print(
            f"Score: "
            f"{first['score']:.3f}"
        )

        print()

        print(
            "Top-1:",
            "✅ PASS"
            if top1_match
            else "❌ FAIL"
        )

        print(
            "Top-3:",
            "✅ PASS"
            if top3_match
            else "❌ FAIL"
        )


        # -------------------------------------------------
        # SHOW TOP 3
        # -------------------------------------------------

        print(
            "\nTop 3 Retrieved:"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['park']} | "
                f"{result['topic']} | "
                f"{result['score']:.3f}"
            )


        print()


    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------

    total = len(
        eval_questions
    )


    if total > 0:

        top1_accuracy = (
            top1_correct
            / total
        ) * 100

        top3_accuracy = (
            top3_correct
            / total
        ) * 100

    else:

        top1_accuracy = 0
        top3_accuracy = 0


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal Questions: "
        f"{total}"
    )


    print(
        f"\nTop-1 Accuracy: "
        f"{top1_correct}/{total} "
        f"({top1_accuracy:.1f}%)"
    )


    print(
        f"Top-3 Accuracy: "
        f"{top3_correct}/{total} "
        f"({top3_accuracy:.1f}%)"
    )


    # -----------------------------------------------------
    # FAILED QUESTIONS
    # -----------------------------------------------------

    if failed_questions:

        print(
            "\n" + "=" * 70
        )

        print(
            "TOP-1 FAILURES"
        )

        print(
            "=" * 70
        )


        for failure in failed_questions:

            print()

            print(
                "Question:",
                failure.get(
                    "question"
                )
            )

            if "expected_park" in failure:

                print(
                    "Expected:",
                    failure[
                        "expected_park"
                    ],
                    "/",
                    failure[
                        "expected_topics"
                    ]
                )

                print(
                    "Retrieved:",
                    failure[
                        "retrieved_park"
                    ],
                    "/",
                    failure[
                        "retrieved_topic"
                    ]
                )

                print(
                    "Score:",
                    f"{failure['score']:.3f}"
                )

            print(
                "-" * 50
            )

    else:

        print(
            "\n🎉 All Top-1 "
            "retrieval tests passed!"
        )