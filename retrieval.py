import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Highest out-of-scope score = 0.4251
# Lowest in-scope score = 0.4862
MIN_SCORE = 0.45
def load_policies():
    with open("fraud_policies.json", "r", encoding="utf-8") as file:
        return json.load(file)

def build_retriever():
    policies = load_policies()
    texts = [policy["text"] for policy in policies]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return policies, vectorizer, matrix

POLICIES, VECTORIZER, POLICY_MATRIX = build_retriever()

def retrieve(query):
    """
    Retrieve the highest-scoring fraud policy.
    Returns:
        dict containing policy information and similarity score,
        or None if the score is below MIN_SCORE.
    """
    query_vector = VECTORIZER.transform([query])
    scores = cosine_similarity(
        query_vector,
        POLICY_MATRIX
    )[0]
    best_index = scores.argmax()
    best_score = float(scores[best_index])
    if best_score < MIN_SCORE:
        return None
    policy = POLICIES[best_index]
    return {
        "policy_id": policy["policy_id"],
        "title": policy["title"],
        "text": policy["text"],
        "score": best_score,
    }

def main():
    with open("retrieval_queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)

    print("\nPART C — POLICY RETRIEVAL")
    print(f"MIN_SCORE: {MIN_SCORE}")
    print()
    print(
        f"{'Query':55} "
        f"{'Expected':10} "
        f"{'Retrieved':10} "
        f"{'Score':>8} "
        f"{'PASS':>6}"
    )
    print("-" * 100)
    for item in queries:
        query = item["query"]
        expected = item["expected_policy"]
        result = retrieve(query)
        if result is None:
            retrieved = None
            score = 0.0
        else:
            retrieved = result["policy_id"]
            score = result["score"]

        passed = retrieved == expected
        print(
            f"{query[:55]:55} "
            f"{str(expected):10} "
            f"{str(retrieved):10} "
            f"{score:8.4f} "
            f"{str(passed):>6}"
        )

if __name__ == "__main__":
    main()
