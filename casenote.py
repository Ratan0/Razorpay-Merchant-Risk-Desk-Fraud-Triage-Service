import json
from mock_llm import generate, reset
from retrieval import retrieve

TEAM_BY_POLICY = {
    "P1": "card_risk",
    "P2": "upi_risk",
    "P3": "customer_care",
    "P4": "merchant_ops",
    "P5": "merchant_ops",
    "P6": "customer_care",
}

ALLOWED_PRIORITIES = {
    "low",
    "medium",
    "high",
}

ALLOWED_TEAMS = {
    "card_risk",
    "upi_risk",
    "customer_care",
    "merchant_ops",
}

ALLOWED_ACTIONS = {
    "approve",
    "hold_for_review",
    "block_and_notify",
}

REQUIRED_KEYS = {
    "summary",
    "priority",
    "recommended_team",
    "recommended_action",
    "policy_id",
}

def build_prompt(
    txn_id,
    fraud_probability,
    amount_inr,
    policy,
):
    """
    Build a structured prompt for the language model.
    """

    prompt = f"""
ROLE:
You are a fraud-risk case-note assistant for a merchant risk desk.

TASK:
Create a concise case note using only the supplied fraud policy.

CONSTRAINTS:
- Follow the supplied policy.
- Do not invent policy IDs.
- Return only a JSON object.
- All five required fields must be strings.
- recommended_action must be one of:
  approve, hold_for_review, block_and_notify.

FORMAT:
Return exactly these JSON fields:
summary
priority
recommended_team
recommended_action
policy_id

POLICY_ID: {policy["policy_id"]}
POLICY_TITLE: {policy["title"]}
POLICY_TEXT: {policy["text"]}

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy["policy_id"]}
"""

    return prompt.strip()

def validate(
    text,
    expected_policy_id,
):
    """
    Three validation gates.

    Gate 1:
        Valid JSON object.

    Gate 2:
        Exactly the five required keys and
        all values are strings.

    Gate 3:
        Allowed enum values and matching policy ID.
    """

    # Gate 1 — valid JSON object

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False

    if not isinstance(data, dict):
        return False
   
    # Gate 2 — required keys + string values
   
    if set(data.keys()) != REQUIRED_KEYS:
        return False

    if not all(
        isinstance(data[key], str)
        for key in REQUIRED_KEYS
    ):
        return False

    # Gate 3 — enums + policy match
   
    if data["priority"] not in ALLOWED_PRIORITIES:
        return False

    if data["recommended_team"] not in ALLOWED_TEAMS:
        return False

    if data["recommended_action"] not in ALLOWED_ACTIONS:
        return False

    if data["policy_id"] != expected_policy_id:
        return False

    return True


def safe_default(
    policy_id,
    fraud_probability,
):
    """
    Safe fallback when all three LLM attempts fail validation.
    """

    priority = (
        "high"
        if fraud_probability >= 0.8
        else "medium"
    )

    return {
        "summary": (
            "Case note generation failed validation; "
            "manual review is required."
        ),
        "priority": priority,
        "recommended_team": TEAM_BY_POLICY[policy_id],
        "recommended_action": "hold_for_review",
        "policy_id": policy_id,
    }

def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
):
    """
    Retrieve policy, call mock LLM and validate the output.

    Maximum attempts = 3.

    ConnectionError is deliberately propagated.
    """
    policy = retrieve(question)

    # No relevant policy found.
    if policy is None:
        return {
            "case_note": None,
            "attempts": 0,
            "safe_default": False,
            "override": False,
        }

    policy_id = policy["policy_id"]

    prompt = build_prompt(
        txn_id=txn_id,
        fraud_probability=fraud_probability,
        amount_inr=amount_inr,
        policy=policy,
    )

    attempts = 0

    for _ in range(3):
        attempts += 1

        try:
            response = generate(prompt)

        except ConnectionError:
            # Requirement: propagate ConnectionError.
            raise

        if validate(
            response,
            expected_policy_id=policy_id,
        ):
            case_note = json.loads(response)

            override = False

            # Output guardrail:
            # Never allow approve for a high-risk payment.
            if (
                case_note["recommended_action"] == "approve"
                and fraud_probability >= 0.5
            ):
                case_note["recommended_action"] = (
                    "hold_for_review"
                )
                override = True

            return {
                "case_note": case_note,
                "attempts": attempts,
                "safe_default": False,
                "override": override,
            }

    # All three attempts failed

    return {
        "case_note": safe_default(
            policy_id,
            fraud_probability,
        ),
        "attempts": attempts,
        "safe_default": True,
        "override": False,
    }

def main():

    with open(
        "case_scenarios.json",
        "r",
        encoding="utf-8",
    ) as file:
        scenarios = json.load(file)

    reset()

    print("\nPART C — CASE NOTE VALIDATION")
    print()

    print(
        f"{'TXN_ID':10} "
        f"{'Policy':8} "
        f"{'Attempts':10} "
        f"{'Safe Default':13} "
        f"{'Override':10} "
        f"{'Action':20}"
    )

    print("-" * 80)

    total_calls = 0
    first_call_valid = 0
    recovered = 0
    safe_defaults = 0
    overrides = 0

    for scenario in scenarios:

        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"],
        )

        attempts = result["attempts"]
        safe_default_used = result["safe_default"]
        override_used = result["override"]

        total_calls += attempts

        if attempts == 1:
            first_call_valid += 1
        elif attempts > 1 and not safe_default_used:
            recovered += 1

        if safe_default_used:
            safe_defaults += 1

        if override_used:
            overrides += 1

        note = result["case_note"]

        if note is None:
            policy_id = None
            action = None
        else:
            policy_id = note["policy_id"]
            action = note["recommended_action"]

        print(
            f"{scenario['txn_id']:10} "
            f"{str(policy_id):8} "
            f"{attempts:<10} "
            f"{str(safe_default_used):13} "
            f"{str(override_used):10} "
            f"{str(action):20}"
        )

    print()
    print("Part C summary")
    print("----------------")
    print("Total LLM calls:", total_calls)
    print("First-call valid:", first_call_valid)
    print("Recovered:", recovered)
    print("Safe defaults:", safe_defaults)
    print("Overrides:", overrides)

if __name__ == "__main__":
    main()
