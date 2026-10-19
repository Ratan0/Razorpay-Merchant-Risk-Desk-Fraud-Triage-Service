import json
import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from prepare import clean_transactions, add_features
from train import build_final_model
from casenote import run_case

# FASTAPI APP

app = FastAPI(
    title="Fraud Triage Service",
    version="1.0.0",
)

# MODEL
MODEL_ARTIFACT = build_final_model()

MODEL = MODEL_ARTIFACT["model"]
MODEL_COLUMNS = MODEL_ARTIFACT["columns"]
TRAINING_MEDIAN = MODEL_ARTIFACT["median"]
THRESHOLD = MODEL_ARTIFACT["threshold"]
# VALID VALUES
VALID_CATEGORIES = {
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
}
VALID_CITIES = {
    "Pune",
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Kolkata",
}
VALID_PAYMENT_MODES = {
    "UPI",
    "Card",
    "NetBanking",
    "Wallet",
}
# METRICS
METRICS = {
    "requests_total": 0,
    "post_requests": 0,
    "errors_total": 0,
    "latency_total_ms": 0.0,
    "latency_count": 0,
}
# LOGGING
logger = logging.getLogger("fraud_triage")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    "service.log",
    mode="a",
    encoding="utf-8",
)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# PYDANTIC MODELS
class Txn(BaseModel):
    txn_id: str
    customer_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int

class CaseIn(Txn):
    fraud_probability: float | None = None
    question: str
# MIDDLEWARE
@app.middleware("http")
async def observability_middleware(
    request: Request,
    call_next,
):
    start = time.perf_counter()
    response = None
    error = None
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error = exc
        raise
    finally:
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000
        METRICS["requests_total"] += 1
        METRICS["latency_total_ms"] += elapsed_ms
        METRICS["latency_count"] += 1
        if request.method == "POST":
            METRICS["post_requests"] += 1
            if error is not None:
                METRICS["errors_total"] += 1
                logger.error(
                    "%s %s status=500 latency_ms=%.2f",
                    request.method,
                    request.url.path,
                    elapsed_ms,
                )
            elif response is not None:
                status = response.status_code
                if status >= 500:
                    level = logging.ERROR
                    METRICS["errors_total"] += 1
                elif status >= 400:
                    level = logging.WARNING
                    METRICS["errors_total"] += 1
                else:
                    level = logging.INFO
                logger.log(
                    level,
                    "%s %s status=%s latency_ms=%.2f",
                    request.method,
                    request.url.path,
                    status,
                    elapsed_ms,
                )
# DOMAIN VALIDATION
def validate_transaction_domain(txn: Txn):
    """
    Domain/range violations must return HTTP 400.
    Schema/type violations are handled automatically by FastAPI
    and return HTTP 422.
    """

    if not (
        1 <= txn.amount_inr <= 1_000_000
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "amount_inr must be between 1 and 1,000,000",
                "field": "amount_inr",
            },
        )

    if not (
        0 <= txn.hour_of_day <= 23
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "hour_of_day must be between 0 and 23",
                "field": "hour_of_day",
            },
        )

    if txn.merchant_category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid merchant_category",
                "field": "merchant_category",
            },
        )

    if txn.city not in VALID_CITIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid city",
                "field": "city",
            },
        )

    if txn.payment_mode not in VALID_PAYMENT_MODES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid payment_mode",
                "field": "payment_mode",
            },
        )
    if txn.txn_count_last_24h < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "txn_count_last_24h cannot be negative",
                "field": "txn_count_last_24h",
            },
        )
    if txn.avg_amount_30d_inr <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "avg_amount_30d_inr must be positive",
                "field": "avg_amount_30d_inr",
            },
        )
    if txn.distance_from_home_km is not None and txn.distance_from_home_km < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "distance_from_home_km cannot be negative",
                "field": "distance_from_home_km",
            },
        )
    if txn.is_new_device not in {0, 1}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "is_new_device must be 0 or 1",
                "field": "is_new_device",
            },
        )
# MODEL INPUT PREPARATION
def transaction_to_features(txn: Txn):
    """
    Convert one API transaction into the exact feature structure
    used by the trained model.
    """
    row = {
        "txn_id": txn.txn_id,
        "customer_id": txn.customer_id,
        "merchant_category": txn.merchant_category,
        "city": txn.city,
        "payment_mode": txn.payment_mode,
        "amount_inr": txn.amount_inr,
        "hour_of_day": txn.hour_of_day,
        "txn_count_last_24h": txn.txn_count_last_24h,
        "avg_amount_30d_inr": txn.avg_amount_30d_inr,
        "distance_from_home_km": txn.distance_from_home_km,
        "is_new_device": txn.is_new_device,
        "account_age_days": txn.account_age_days,
    }
    df = pd.DataFrame([row])
    # Part A feature engineering
    df = add_features(df)
    # Remove identifiers.
    df = df.drop(
        columns=[
            "txn_id",
            "customer_id",
        ],
        errors="ignore",
    )
    # One-hot encoding.
    df = pd.get_dummies(
        df,
        columns=[
            "merchant_category",
            "city",
            "payment_mode",
        ],
        dtype=int,
    )
    df = df.reindex(
        columns=MODEL_COLUMNS,
        fill_value=0,
    )
    if "distance_from_home_km" in df.columns:
        df["distance_from_home_km"] = (
            df["distance_from_home_km"]
            .fillna(TRAINING_MEDIAN)
        )
    df = df.fillna(0)
    return df
# SCORE TRANSACTION
def score_transaction(txn):
    X = transaction_to_features(txn)

    probability = float(MODEL.predict_proba(X)[0, 1])

    # Normal ML decision
    model_flagged = probability >= THRESHOLD

    # Hard guardrail
    is_big_new_device = (
        txn.amount_inr >= 100000
        and txn.is_new_device == 1
    )

    if is_big_new_device:
        # IMPORTANT:
        # This is a hard hold, not an ML fraud flag.
        flagged = False
        decision = "hold_for_review"
        human_review = True

    elif model_flagged:
        flagged = True
        decision = "hold_for_review"
        human_review = True

    else:
        flagged = False
        decision = "approve"
        human_review = False

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
    }

# GET /health

@app.get("/health")
def health():
    return {
        "status": "ok"
    }
# POST /score
@app.post("/score")
def score(txn: Txn):
    validate_transaction_domain(txn)
    return score_transaction(txn)
# POST /case
@app.post("/case")
def case(txn: CaseIn):
    validate_transaction_domain(txn)

    result = score_transaction(txn)
    # First score the transaction.
    score_result = score_transaction(
        txn
    )

    flagged = score_result["flagged"]
    
    # Benign transaction
    
    if not flagged:

        return {
            **score_result,
            "case_note": None,
            "attempts": 0,
            "safe_default": False,
        }

    fraud_probability = score_result[
        "fraud_probability"
    ]

    try:

        result = run_case(
            txn_id=txn.txn_id,
            fraud_probability=fraud_probability,
            amount_inr=txn.amount_inr,
            question=txn.question,
        )

    except ConnectionError as exc:

        raise HTTPException(
            status_code=503,
            detail={
                "error": str(exc)
            },
        )

    # Case-level guardrail
    case_note = result["case_note"]
    override = result.get(
        "override",
        False,
    )

    if case_note is not None:

        if (
            case_note["recommended_action"]
            == "approve"
        ):
            case_note[
                "recommended_action"
            ] = "hold_for_review"

            override = True
    return {
        **score_result,
        "case_note": case_note,
        "attempts": result["attempts"],
        "safe_default": result["safe_default"],
        "override": override,
    }
# GET /metrics
@app.get("/metrics")
def metrics():

    if METRICS["latency_count"] > 0:
        avg_latency = (
            METRICS["latency_total_ms"]
            / METRICS["latency_count"]
        )
    else:
        avg_latency = 0.0

    return {
        "requests_total": METRICS["requests_total"],
        "post_requests": METRICS["post_requests"],
        "errors_total": METRICS["errors_total"],
        "average_latency_ms": round(
            avg_latency,
            3,
        ),
    }
