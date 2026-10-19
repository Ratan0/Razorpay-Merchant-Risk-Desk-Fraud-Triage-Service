import requests
BASE_URL = "http://127.0.0.1:8000"

def print_result(
    test_id,
    response,
):
    print(
        f"{test_id:4} "
        f"status={response.status_code} "
        f"{response.json()}"
    )

def main():
    print("\nPART D — SMOKE TEST")
    print("=" * 100)
    # Q1 — Health
    response = requests.get(
        f"{BASE_URL}/health"
    )

    print_result(
        "Q1",
        response,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # Common transaction
    benign = {
        "txn_id": "T800001",
        "customer_id": "C800001",
        "merchant_category": "Grocery",
        "city": "Pune",
        "payment_mode": "UPI",
        "amount_inr": 500,
        "hour_of_day": 14,
        "txn_count_last_24h": 1,
        "avg_amount_30d_inr": 700,
        "distance_from_home_km": 5,
        "is_new_device": 0,
        "account_age_days": 500,
    }
    # Q2 — Benign score
    response = requests.post(
        f"{BASE_URL}/score",
        json=benign,
    )

    print_result(
        "Q2",
        response,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["flagged"] is False
    assert result["decision"] == "approve"
    assert result["human_review"] is False

    # Q3 — Risky score
    risky = {
        **benign,
        "txn_id": "T800002",
        "amount_inr": 6000,
        "hour_of_day": 2,
        "txn_count_last_24h": 10,
        "distance_from_home_km": 150,
        "is_new_device": 1,
        "account_age_days": 10,
    }

    response = requests.post(
        f"{BASE_URL}/score",
        json=risky,
    )

    print_result(
        "Q3",
        response,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["flagged"] is True
    assert result["decision"] == "hold_for_review"
    assert result["human_review"] is True
 # Q4 — Big payment hard guardrail
    big = {
        **benign,
        "txn_id": "T800003",
        "amount_inr": 120000,
        "is_new_device": 1,
    }

    response = requests.post(
        f"{BASE_URL}/score",
        json=big,
    )

    print_result(
        "Q4",
        response,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["flagged"] is False
    assert result["decision"] == "hold_for_review"
    assert result["human_review"] is True
    # Q5 — Invalid amount
    invalid_amount = {
        **benign,
        "txn_id": "T800004",
        "amount_inr": -5,
    }

    response = requests.post(
        f"{BASE_URL}/score",
        json=invalid_amount,
    )

    print_result(
        "Q5",
        response,
    )

    assert response.status_code == 400

    body = response.json()

    assert body["detail"]["field"] == "amount_inr"
    # Q6 — Missing required field
    missing_hour = {
        **benign,
        "txn_id": "T800005",
    }

    del missing_hour["hour_of_day"]

    response = requests.post(
        f"{BASE_URL}/score",
        json=missing_hour,
    )

    print_result(
        "Q6",
        response,
    )

    assert response.status_code == 422
    # Q7 — Risky case
    case_risky = {
        **risky,
        "txn_id": "T900002",
        "question": (
            "more than 6 UPI payments in 24 hours "
            "velocity alert"
        ),
    }

    response = requests.post(
        f"{BASE_URL}/case",
        json=case_risky,
    )

    print_result(
        "Q7",
        response,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["flagged"] is True
    assert result["attempts"] == 2
    assert result["safe_default"] is False
    assert result["case_note"]["policy_id"] == "P2"
    assert result["human_review"] is True
    # Q8 — Three failed attempts -> safe default
    case_failed = {
        **risky,
        "txn_id": "T900005",
        "amount_inr": 41000,
        "merchant_category": "Electronics",
        "account_age_days": 10,
        "question": (
            "electronics merchant payment on an "
            "account younger than 30 days"
        ),
    }

    response = requests.post(
        f"{BASE_URL}/case",
        json=case_failed,
    )

    print_result(
        "Q8",
        response,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["attempts"] == 3
    assert result["safe_default"] is True
    assert result["decision"] == "hold_for_review"
    assert result["human_review"] is True
    # Q9 — Simulated LLM outage
    outage = {
        **risky,
        "txn_id": "T900099",
        "question": (
            "card payment above INR 50000 "
            "from a new device"
        ),
    }

    response = requests.post(
        f"{BASE_URL}/case",
        json=outage,
    )
    print_result(
        "Q9",
        response,
    )
    assert response.status_code == 503
    benign_case = {
        **benign,
        "txn_id": "T800001",
        "question": (
            "card payment above INR 50000 "
            "from a new device"
        ),
    }
    response = requests.post(
        f"{BASE_URL}/case",
        json=benign_case,
    )
    print_result(
        "Q10",
        response,
    )
    assert response.status_code == 200
    result = response.json()
    assert result["case_note"] is None
    assert result["attempts"] == 0
    assert result["decision"] == "approve"
    assert result["human_review"] is False

    # Q11 — Flagged but out-of-scope policy question
    out_of_scope = {
        **risky,
        "txn_id": "T800006",
        "question": "what is the cafeteria menu today",
    }
    response = requests.post(
        f"{BASE_URL}/case",
        json=out_of_scope,
    )
    print_result(
        "Q11",
        response,
    )
    assert response.status_code == 200
    result = response.json()
    assert result["case_note"] is None
    assert result["attempts"] == 0
    assert result["decision"] == "hold_for_review"
    assert result["human_review"] is True
    print()
    print("=" * 100)
    print("11/11 SMOKE TESTS PASSED")

if __name__ == "__main__":
    main()
