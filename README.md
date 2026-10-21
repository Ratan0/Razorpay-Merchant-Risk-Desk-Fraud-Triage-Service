# Razorpay Merchant Risk Desk — Fraud Triage Service
An end-to-end fraud triage service built for a one-day capstone assignment using synthetic payment data.
> **Note:** All data and policies in this project are synthetic and fictional.
## Overview
The project is divided into four connected parts:

- **Part A — Data Foundation:** audit, clean, and feature-engineer transaction data.
- **Part B — Fraud Model:** train, compare, tune, and threshold a fraud classifier.
- **Part C — Policy-Grounded Case Notes:** retrieve fraud policies and generate validated structured case notes.
- **Part D — Triage Service:** expose the complete pipeline through a FastAPI service with validation, guardrails, logging, and metrics.

## Project Structure

```text
razorpay-merchant-risk-desk/
├── generate_data.py
├── mock_llm.py
├── transactions_raw.csv
├── fraud_policies.json
├── retrieval_queries.json
├── case_scenarios.json
├── prepare.py
├── train.py
├── retrieval.py
├── casenote.py
├── app.py
├── smoke_test.py
├── requirements.txt
├── .gitignore
├── service.log
└── README.md
```

## Technology Stack

- Python 3.11+
- pandas
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Pydantic
- Requests
- TF-IDF and cosine similarity
- Logistic Regression
- Random Forest

# Install karne ka command
G:\project ke andar virtual environment activate karke:
cd /d G:\project
.venv\Scripts\activate
pip install -r Requirement.txt

or do it indevidualy.

python -c "import fastapi, uvicorn, pydantic, requests, pandas, sklearn; print('All requirements OK')"

## Part A — Data Preparation

`prepare.py` provides:

```python
clean_transactions(raw_df)
add_features(df)
```

Cleaning includes:

- Exact duplicate removal
- Merchant-category normalization
- City normalization
- Missing amount removal
- Amount range validation

Feature engineering includes:

- `amount_vs_avg_ratio`
- `is_night`
- `high_value`

## Part B — Fraud Model

The model pipeline uses a 60/20/20 stratified train/validation/test split with `random_state=42`.

Models compared include:

- Plain Logistic Regression
- Class-balanced Logistic Regression
- Over-sampled Logistic Regression
- Random Forest

Random Forest tuning uses 5-fold `StratifiedKFold` cross-validation.

The alert threshold is selected from 0.1–0.9 by choosing the highest threshold with validation recall of at least 0.65.

Selected threshold:

```text
0.4
```

`build_final_model()` returns:

```python
{
    "model": ...,
    "columns": ...,
    "median": ...,
    "threshold": ...
}
```

### Reference checkpoints

```text
Train / Validation / Test: 3551 / 1184 / 1184
Train fraud:               220
Validation fraud:           73
Test fraud:                 73
Training distance median:  12.6
Selected threshold:          0.4
```

Reference tuned Random Forest:

```text
max_depth = None
min_samples_leaf = 5
```

## Part C — Policy-Grounded Case Notes

`retrieval.py` uses the default `TfidfVectorizer` and cosine similarity to retrieve the most relevant fraud policy.

`casenote.py` provides:

```python
build_prompt(...)
validate(...)
run_case(...)
```

Case notes contain exactly five fields:

```text
summary
priority
recommended_team
recommended_action
policy_id
```

The pipeline supports:

- Policy retrieval
- JSON validation
- Required-key validation
- Allowed-value validation
- Policy-ID matching
- Up to three attempts
- Safe fallback
- Connection-error propagation
- Approval guardrails

## Part D — FastAPI Service

The service is implemented in `app.py`.

Start the API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/score` | Score a transaction |
| POST | `/case` | Generate a policy-grounded case note |
| GET | `/metrics` | Service metrics |

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

## Validation and Guardrails

Schema/type errors return:

```text
HTTP 422
```

Domain/range violations return:

```text
HTTP 400
```

Examples include invalid:

- `amount_inr`
- `hour_of_day`
- `merchant_category`
- `city`
- `payment_mode`

### Big-payment guardrail

Payments of at least ₹100,000 from a new device are held for human review.

This guardrail is separate from the ML fraud flag. For the BIG smoke-test transaction:

```text
flagged       = false
decision      = hold_for_review
human_review  = true
```

### LLM outage

A mock LLM `ConnectionError` is mapped to:

```text
HTTP 503
```

## Running the Project

### 1. Create the virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scriptsctivate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Part A

```bash
python prepare.py
```

### 4. Run Part B

```bash
python train.py
```

### 5. Run Part C

```bash
python retrieval.py
python casenote.py
```

### 6. Start the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 7. Run the smoke test

Keep Uvicorn running and use a second terminal:

```bash
python smoke_test.py
```

## Smoke Test

The Part D smoke test contains 11 scenarios covering:

- Health check
- Benign transaction scoring
- Risky transaction scoring
- Big-payment guardrail
- Invalid amount
- Missing required field
- Case-note retry
- Safe fallback
- LLM outage
- Benign case handling
- Out-of-scope case handling

Final verified result:

```text
11/11 SMOKE TESTS PASSED
```

## Logging and Metrics

The service writes request logs to:

```text
service.log
```

Middleware records:

- Total requests
- POST requests
- Errors
- Average latency

## Final Submission Checklist

- [x] Part A completed
- [x] Part B completed
- [x] Part C completed
- [x] Part D completed
- [x] `/health` working
- [x] `/score` working
- [x] `/case` working
- [x] `/metrics` working
- [x] Domain validation implemented
- [x] Score guardrail implemented
- [x] Case-note guardrail implemented
- [x] HTTP 503 LLM error handling implemented
- [x] Smoke test: **11/11 passed**
- [ ] Git installed and configured
- [ ] GitHub repository connected
- [ ] Final project pushed to `main`
- [ ] Clean-clone smoke test verified

## Repository

The final repository should contain the complete project and point to the `main` branch.

---

**Project:** Razorpay Merchant Risk Desk — Fraud Triage Service  
**Data:** Synthetic / Fictional  
**Status:** Part D smoke test passed — **11/11**
