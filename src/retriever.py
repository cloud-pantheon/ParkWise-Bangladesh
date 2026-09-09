import numpy as np
from sentence_transformers import SentenceTransformer

from dataset_loader import load_dataset


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

def load_all_records():
    return load_dataset()


# ---------------------------------------------------------
# BUILD EMBEDDINGS
# ---------------------------------------------------------

def build_embeddings(records, model):

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


# ---------------------------------------------------------
# PARK DETECTION
# ---------------------------------------------------------

def detect_park(query):

    query_lower = query.lower()

    if "lawachara" in query_lower:
        return "Lawachara National Park"

    if "satchari" in query_lower:
        return "Satchari National Park"

    if "bhawal" in query_lower:
        return "Bhawal National Park"

    return None


# ---------------------------------------------------------
# QUERY EXPANSION
# ---------------------------------------------------------

def expand_query(query):

    query_lower = query.lower()

    extra_terms = []

    # Wildlife
    if (
        "wildlife" in query_lower
        or "animal" in query_lower
        or "animals" in query_lower
        or "bird" in query_lower
        or "birds" in query_lower
    ):
        extra_terms.extend([
            "wildlife",
            "animals",
            "mammals",
            "birds",
            "reptiles",
            "amphibians",
            "biodiversity"
        ])

    # Forest
    if (
        "forest" in query_lower
        or "tree" in query_lower
        or "trees" in query_lower
        or "vegetation" in query_lower
    ):
        extra_terms.extend([
            "forest type",
            "vegetation",
            "sal forest",
            "evergreen forest",
            "semi-evergreen forest"
        ])

    # Facilities
    if (
        "facility" in query_lower
        or "facilities" in query_lower
        or "visitor" in query_lower
        or "tourist" in query_lower
        or "tourism" in query_lower
    ):
        extra_terms.extend([
            "visitor facilities",
            "tourism",
            "ecotourism",
            "trails",
            "information center",
            "picnic",
            "toilets"
        ])

    # Location
    if (
        "where" in query_lower
        or "location" in query_lower
        or "located" in query_lower
    ):
        extra_terms.extend([
            "location",
            "district",
            "upazila",
            "Bangladesh"
        ])

    # Access
    if (
        "reach" in query_lower
        or "get to" in query_lower
        or "travel" in query_lower
        or "go to" in query_lower
        or "access" in query_lower
    ):
        extra_terms.extend([
            "access",
            "road",
            "transport",
            "travel",
            "route"
        ])

    # Management
    if (
        "management" in query_lower
        or "conservation" in query_lower
        or "protect" in query_lower
    ):
        extra_terms.extend([
            "management",
            "conservation",
            "co-management",
            "stakeholders",
            "biodiversity protection"
        ])

    # Community
    if (
        "community" in query_lower
        or "communities" in query_lower
        or "local people" in query_lower
        or "indigenous" in query_lower
    ):
        extra_terms.extend([
            "communities",
            "local people",
            "indigenous communities",
            "stakeholders",
            "livelihoods"
        ])

    # Threats
    if (
        "threat" in query_lower
        or "threats" in query_lower
        or "problem" in query_lower
        or "problems" in query_lower
        or "danger" in query_lower
    ):
        extra_terms.extend([
            "threats",
            "encroachment",
            "industrialization",
            "urbanization",
            "conservation problems"
        ])

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

def search(
    query,
    records,
    embeddings,
    model,
    top_k=5
):

    query_lower = query.lower()

    # Detect park mentioned in question
    detected_park = detect_park(query)

    # Expand the query
    expanded_query = expand_query(query)

    # Turn query into embedding
    query_embedding = model.encode(
        expanded_query,
        normalize_embeddings=True
    )

    # Cosine similarity
    scores = np.dot(
        embeddings,
        query_embedding
    )

    results = []

    for index, score in enumerate(scores):

        record = records[index]

        # -------------------------------------------------
        # PARK METADATA FILTER
        # -------------------------------------------------

        if (
            detected_park
            and record["park"] != detected_park
        ):
            continue

        adjusted_score = float(score)

        topic = record["topic"].lower()
        text = record["text"].lower()


        # -------------------------------------------------
        # TOPIC BOOSTS
        # -------------------------------------------------

        # Wildlife
        if (
            "wildlife" in query_lower
            or "animal" in query_lower
            or "animals" in query_lower
            or "bird" in query_lower
            or "birds" in query_lower
        ):

            if "wildlife" in topic:
                adjusted_score += 0.30

            elif "birds" in topic:
                adjusted_score += 0.25

            elif "biodiversity" in topic:
                adjusted_score += 0.20


        # Location
        if (
            "where" in query_lower
            or "location" in query_lower
            or "located" in query_lower
        ):

            if "identity_location" in topic:
                adjusted_score += 0.30

            elif "access" in topic:
                adjusted_score += 0.10


        # Access
        if (
            "reach" in query_lower
            or "get to" in query_lower
            or "travel" in query_lower
            or "go to" in query_lower
            or "access" in query_lower
        ):

            if "access" in topic:
                adjusted_score += 0.30


        # Facilities
        if (
            "facility" in query_lower
            or "facilities" in query_lower
        ):

            if "facilities" in topic:
                adjusted_score += 0.35

            elif "visitor" in topic:
                adjusted_score += 0.15

            elif "ecotourism" in topic:
                adjusted_score += 0.10

            elif "trails_guides" in topic:
                adjusted_score += 0.05


        # Forest type
        if (
            "forest" in query_lower
            or "vegetation" in query_lower
            or "tree" in query_lower
            or "trees" in query_lower
        ):

            if "forest_type" in topic:
                adjusted_score += 0.30

            elif "plants" in topic:
                adjusted_score += 0.10


        # Management
        if (
            "management" in query_lower
            or "manage" in query_lower
        ):

            if "management" in topic:
                adjusted_score += 0.30

            elif "co_management" in topic:
                adjusted_score += 0.25


        # Conservation
        if (
            "conservation" in query_lower
            or "protect" in query_lower
            or "protection" in query_lower
        ):

            if "management_goals" in topic:
                adjusted_score += 0.25

            elif "co_management" in topic:
                adjusted_score += 0.20

            elif "land_use_threats" in topic:
                adjusted_score += 0.15


        # Communities
        if (
            "community" in query_lower
            or "communities" in query_lower
            or "local people" in query_lower
            or "indigenous" in query_lower
        ):

            if "communities" in topic:
                adjusted_score += 0.30

            elif "livelihoods" in topic:
                adjusted_score += 0.20


        # Threats
        if (
            "threat" in query_lower
            or "threats" in query_lower
            or "problem" in query_lower
            or "problems" in query_lower
        ):

            if "land_use_threats" in topic:
                adjusted_score += 0.30


        # Ecotourism
        if (
            "ecotourism" in query_lower
            or "tourism" in query_lower
        ):

            if "ecotourism" in topic:
                adjusted_score += 0.30

            elif "visitor_use" in topic:
                adjusted_score += 0.15


        # Biodiversity monitoring
        if (
            "monitor" in query_lower
            or "monitoring" in query_lower
        ):

            if "monitoring" in topic:
                adjusted_score += 0.35


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        results.append({
            "score": adjusted_score,

            "id":
                record["id"],

            "park":
                record["park"],

            "topic":
                record["topic"],

            "text":
                record["text"],

            "source_title":
                record["source_title"],

            "source_url":
                record["source_url"],

            "source_year":
                record["source_year"],

            "source_type":
                record["source_type"],

            "freshness_note":
                record["freshness_note"]
        })


    # Sort highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ---------------------------------------------------------
# TERMINAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Loading Bangladesh Parks dataset..."
    )

    records = load_all_records()

    print(
        f"Loaded {len(records)} records."
    )

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Creating embeddings..."
    )

    embeddings = build_embeddings(
        records,
        model
    )

    print(
        "\nBangladesh Parks Retriever Ready!"
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        query = input(
            "\nAsk a question: "
        )

        if query.lower() == "exit":

            print(
                "Goodbye!"
            )

            break


        results = search(
            query,
            records,
            embeddings,
            model,
            top_k=5
        )


        print(
            "\nTop Results:"
        )

        print(
            "=" * 70
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Score: {result['score']:.3f}"
            )

            print(
                f"Park: {result['park']}"
            )

            print(
                f"Topic: {result['topic']}"
            )

            print(
                f"Source: {result['source_title']}"
            )

            print()

            print(
                result["text"]
            )

            print(
                "\n" + "-" * 70
            )