import json
from pathlib import Path


def load_dataset():

    data_path = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "data"
        / "bd_parks_rag_corpus.jsonl"
    )

    records = []

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

    return records


if __name__ == "__main__":

    records = load_dataset()

    print(
        f"Loaded {len(records)} records."
    )

    print(
        "\nExample Record:"
    )

    print(
        records[0]
    )