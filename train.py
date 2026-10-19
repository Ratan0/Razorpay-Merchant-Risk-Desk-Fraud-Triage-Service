"""Train, evaluate, tune, and package the fraud model for Part B."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import clean_transactions, add_features


DATA_PATH = Path(__file__).with_name("transactions_raw.csv")

# 1. Prepare data

def prepare_model_data():
    """Load, clean, feature-engineer and one-hot encode the dataset."""

    raw_df = pd.read_csv(DATA_PATH)

    cleaned = clean_transactions(raw_df)
    featured = add_features(cleaned)

    target = "is_fraud"

    excluded = [
        "txn_id",
        "customer_id",
        "chargeback_filed",
        target,
    ]

    X = featured.drop(columns=excluded)
    y = featured[target]

    categorical_columns = [
        "merchant_category",
        "city",
        "payment_mode",
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int,
    )

    # Required column order from the assignment.
    numeric_columns = [
        "amount_inr",
        "hour_of_day",
        "txn_count_last_24h",
        "avg_amount_30d_inr",
        "distance_from_home_km",
        "is_new_device",
        "account_age_days",
        "amount_vs_avg_ratio",
        "is_night",
        "high_value",
    ]

    one_hot_columns = sorted(
        [
            column
            for column in X.columns
            if column not in numeric_columns
        ]
    )

    columns = numeric_columns + one_hot_columns

    X = X[columns]

    return X, y, columns


# 2. Split data

def split_data(X, y):
    """Create the required stratified 60/20/20 split."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    return (
        X_train.copy(),
        X_val.copy(),
        X_test.copy(),
        y_train.copy(),
        y_val.copy(),
        y_test.copy(),
    )

# 3. Fill missing distance

def fill_distance(
    X_train,
    X_val,
    X_test,
):
    """Fill distance using ONLY the training median."""

    median = X_train["distance_from_home_km"].median()

    X_train["distance_from_home_km"] = (
        X_train["distance_from_home_km"]
        .fillna(median)
    )

    X_val["distance_from_home_km"] = (
        X_val["distance_from_home_km"]
        .fillna(median)
    )

    X_test["distance_from_home_km"] = (
        X_test["distance_from_home_km"]
        .fillna(median)
    )

    return X_train, X_val, X_test, median

# 4. Evaluation helper

def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate a probabilistic classifier at a given threshold."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    return precision, recall, f1

# 5. Logistic Regression models

def build_logistic(class_weight=None):
    """Create StandardScaler + LogisticRegression pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )

# 6. Random oversampling

def random_oversample(X_train, y_train):
    """
    Randomly oversample the minority fraud class
    until both classes have equal size.
    """

    train_df = X_train.copy()
    train_df["target"] = y_train.to_numpy()

    majority = train_df[
        train_df["target"] == 0
    ]

    minority = train_df[
        train_df["target"] == 1
    ]

    minority_oversampled = minority.sample(
        n=len(majority),
        replace=True,
        random_state=42,
    )

    balanced = pd.concat(
        [
            majority,
            minority_oversampled,
        ],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    y_balanced = balanced.pop("target")
    X_balanced = balanced

    return X_balanced, y_balanced


# 7. Cross-validation

def cross_validation_models(
    X_train,
    y_train,
):
    """Compare balanced logistic regression and random forest."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    balanced_logistic = build_logistic(
        class_weight="balanced"
    )

    logistic_scores = cross_val_score(
        balanced_logistic,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
    )

    forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
    )

    forest_scores = cross_val_score(
        forest,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
    )

    return (
        balanced_logistic,
        forest,
        logistic_scores,
        forest_scores,
        cv,
    )


# 8. Tune Random Forest
def tune_random_forest(
    X_train,
    y_train,
    cv,
):
    """Tune Random Forest using the required parameter grid."""

    forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
    )

    param_grid = {
        "max_depth": [4,6,8,None],
        "min_samples_leaf": [5,10,20]
    }

    search = GridSearchCV(
        estimator=forest,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(
        X_train,
        y_train,
    )
    best_forest = search.best_estimator_
    return search

# 9. Threshold selection

def find_threshold(model,X_val,y_val):
   
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    thresholds = np.arange(
        0.1,
        1.0,
        0.1,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    1,
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    threshold_table = pd.DataFrame(results)

    valid = threshold_table[
        threshold_table["recall"] >= 0.65
    ]

    if valid.empty:
        raise ValueError(
            "No threshold satisfies recall >= 0.65"
        )

    selected_threshold = float(
        valid["threshold"].max()
    )

    return (
        selected_threshold,
        threshold_table,
    )

# 10. Leakage experiment

def leakage_experiment(
    featured,
    y,
):
    """
    Demonstrate artificial performance improvement
    by incorrectly including chargeback_filed.
    """

    excluded_clean = [
        "txn_id",
        "customer_id",
        "is_fraud",
    ]

    X_leaky = featured.drop(
        columns=excluded_clean
    )

    categorical_columns = [
        "merchant_category",
        "city",
        "payment_mode",
    ]

    X_leaky = pd.get_dummies(
        X_leaky,
        columns=categorical_columns,
        dtype=int,
    )

    columns = sorted(X_leaky.columns)
    X_leaky = X_leaky[columns]

    # Same two-step 60/20/20 split used in Part B.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_leaky,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

        # Impute missing values using ONLY the training median
    distance_median = X_train["distance_from_home_km"].median()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train["distance_from_home_km"] = X_train["distance_from_home_km"].fillna(
        distance_median
    )

    X_val["distance_from_home_km"] = X_val["distance_from_home_km"].fillna(
        distance_median
    )

    X_test["distance_from_home_km"] = X_test["distance_from_home_km"].fillna(
        distance_median
    )

    model = build_logistic()

    model.fit(
        X_train,
        y_train,
    )

    _, _, f1 = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.5,
    )

    return f1

# 11. Final model builder

def build_final_model():
    """
    Build the reusable final model.

    Returns:
        dict containing:
        model
        columns
        median
        threshold
    """

    X, y, columns = prepare_model_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    (
        X_train,
        X_val,
        X_test,
        median,
    ) = fill_distance(
        X_train,
        X_val,
        X_test,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = tune_random_forest(
        X_train,
        y_train,
        cv,
    )

    final_model = search.best_estimator_

    threshold, _ = find_threshold(
        final_model,
        X_val,
        y_val,
    )

    return {
        "model": final_model,
        "columns": columns,
        "median": median,
        "threshold": threshold,
    }

# 12. Main

def main():

    print("=" * 60)
    print("PART B — FRAUD MODEL")
    print("=" * 60)

    # Prepare data
    raw_df = pd.read_csv(DATA_PATH)

    cleaned = clean_transactions(raw_df)
    featured = add_features(cleaned)

    X, y, columns = prepare_model_data()

    print("\nInput columns:", len(columns))
    print("Target rows:", len(y))

    # Split
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    print("\nSplit sizes")
    print(
        f"Train:      {len(X_train)} "
        f"rows, {y_train.sum()} fraud"
    )

    print(
        f"Validation: {len(X_val)} "
        f"rows, {y_val.sum()} fraud"
    )

    print(
        f"Test:       {len(X_test)} "
        f"rows, {y_test.sum()} fraud"
    )

    # Fill missing distance
    (
        X_train,
        X_val,
        X_test,
        median,
    ) = fill_distance(
        X_train,
        X_val,
        X_test,
    )

    print(
        f"\nTraining distance median: "
        f"{median:.1f}"
    )

    # Three imbalance approaches

    plain_model = build_logistic()

    balanced_model = build_logistic(
        class_weight="balanced"
    )

    X_over, y_over = random_oversample(
        X_train,
        y_train,
    )

    oversampled_model = build_logistic()

    plain_model.fit(
        X_train,
        y_train,
    )

    balanced_model.fit(
        X_train,
        y_train,
    )

    oversampled_model.fit(
        X_over,
        y_over,
    )

    print("\nValidation results at threshold 0.5")

    for name, model in [
        ("Plain", plain_model),
        ("Balanced", balanced_model),
        ("Over-sampled", oversampled_model),
    ]:

        precision, recall, f1 = evaluate_model(
            model,
            X_val,
            y_val,
            threshold=0.5,
        )

        print(
            f"{name:15s} "
            f"precision={precision:.3f} "
            f"recall={recall:.3f} "
            f"F1={f1:.3f}"
        )

    print(
        "\nOver-sampled training rows:",
        len(X_over),
    )

    # Cross-validation

    (
        balanced_logistic,
        forest,
        logistic_scores,
        forest_scores,
        cv,
    ) = cross_validation_models(
        X_train,
        y_train,
    )

    print("\n5-fold CV")

    print(
        "Balanced Logistic Regression F1:",
        f"{logistic_scores.mean():.3f}",
    )

    print(
        "Untuned Random Forest F1:",
        f"{forest_scores.mean():.3f}",
    )

    # Random Forest tuning

    search = tune_random_forest(
        X_train,
        y_train,
        cv,
    )

    print("\nBest Random Forest")
    print(
        "Best parameters:",
        search.best_params_,
    )

    print(
        "Best CV F1:",
        f"{search.best_score_:.3f}",
    )

    best_model = search.best_estimator_

    # Threshold
    threshold, threshold_table = find_threshold(
        best_model,
        X_val,
        y_val,
    )

    print("\nThreshold results")

    print(
        threshold_table[
            threshold_table["threshold"].isin(
                [0.4, 0.5, 0.6]
            )
        ].to_string(index=False)
    )

    print(
        f"\nSelected threshold: {threshold:.1f}"
    )

    # Final test 
    precision, recall, f1 = evaluate_model(
        best_model,
        X_test,
        y_test,
        threshold=threshold,
    )

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    print("\nFinal Test Results")
    print(
        f"Precision: {precision:.3f}"
    )
    print(
        f"Recall:    {recall:.3f}"
    )
    print(
        f"F1:        {f1:.3f}"
    )
    print(
        f"Flagged:   {predictions.sum()}"
    )

    print(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    # Leakage experiment
    leaky_f1 = leakage_experiment(
        featured,
        y,
    )

    clean_validation_f1 = f1_score(
        y_val,
        (
            best_model.predict_proba(X_val)[:, 1]
            >= 0.5
        ).astype(int),
    )

    print("\nLeakage Experiment")
    print(
        f"Clean validation F1: "
        f"{clean_validation_f1:.3f}"
    )

    print(
        f"Leaky validation F1: "
        f"{leaky_f1:.3f}"
    )

    # Feature importance 
    importances = pd.Series(
        best_model.named_steps["model"].feature_importances_
        if hasattr(best_model, "named_steps")
        else best_model.feature_importances_,
        index=columns,
    )

    print("\nTop 5 Features")

    print(
        importances
        .sort_values(ascending=False)
        .head(5)
        .to_string()
    )

if __name__ == "__main__":
    main()
