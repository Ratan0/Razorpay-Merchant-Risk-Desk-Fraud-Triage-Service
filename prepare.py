"""Audit, clean, and engineer features for the transaction dataset."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).with_name("transactions_raw.csv")

def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned transactions without modifying the input DataFrame."""
    cleaned = raw_df.drop_duplicates().copy()
    cleaned["merchant_category"] = (
        cleaned["merchant_category"].str.strip().str.title()
    )
    cleaned["city"] = cleaned["city"].str.strip().str.title()
    cleaned = cleaned.dropna(subset=["amount_inr"])
    cleaned = cleaned.loc[cleaned["amount_inr"].between(1, 1_000_000)]
    return cleaned.reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with transaction-derived model features."""
    featured = df.copy()
    featured["amount_vs_avg_ratio"] = (
        featured["amount_inr"] / featured["avg_amount_30d_inr"]
    )
    featured["is_night"] = (featured["hour_of_day"] <= 5).astype(int)
    featured["high_value"] = (featured["amount_inr"] > 50_000).astype(int)
    return featured

def main() -> None:
    raw_df = pd.read_csv(DATA_PATH)
    deduplicated = raw_df.drop_duplicates()
    cleaned = clean_transactions(raw_df)
    featured = add_features(cleaned)

    FEATURED_PATH = Path(__file__).with_name("transactions_clean.csv")
    featured.to_csv(FEATURED_PATH, index=False)

    audit_table = pd.DataFrame(
        [
            ("Rows as read", len(raw_df)),
            ("Columns as read", len(raw_df.columns)),
            ("Exact duplicate rows", int(raw_df.duplicated().sum())),
            (
                "Missing amount_inr after deduplication",
                int(deduplicated["amount_inr"].isna().sum()),
            ),
            (
                "Missing distance_from_home_km after deduplication",
                int(deduplicated["distance_from_home_km"].isna().sum()),
            ),
            (
                "Invalid amount rows after deduplication",
                int(
                    (
                        (deduplicated["amount_inr"] < 1)
                        | (deduplicated["amount_inr"] > 1_000_000)
                    ).sum()
                ),
            ),
            (
                "Raw merchant_category spellings",
                int(raw_df["merchant_category"].nunique()),
            ),
            ("Raw city spellings", int(raw_df["city"].nunique())),
        ],
        columns=["Audit measure", "Count"],
    )
    print("Audit table")
    print(audit_table.to_string(index=False))

    chargeback_summary = (
        cleaned.groupby("chargeback_filed")["is_fraud"]
        .agg(rows="size", fraud_rows="sum")
        .reindex([0, 1], fill_value=0)
    )
    fraud_rows = int(cleaned["is_fraud"].sum())
    fraud_rate = float(cleaned["is_fraud"].mean())
    baseline_accuracy = float((cleaned["is_fraud"] == 0).mean())

    print("\nPart A checkpoints")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Fraud rows: {fraud_rows}")
    print(f"Fraud rate: {fraud_rate:.6%}")
    print(f"Always-not-fraud accuracy: {baseline_accuracy:.6%}")
    print("Always-not-fraud fraud recall: 0.000000%")
    print("Chargeback breakdown (rows, fraud_rows):")
    print(chargeback_summary.to_string())
    print(f"Feature columns added: {', '.join(featured.columns[-3:])}")

if __name__ == "__main__":
    main()
