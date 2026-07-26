# %% [markdown]
# # Phase 2 Ablation Comparison: Original Dataset vs. Dataset v2
#
# This comparison mirrors the existing Phase 1 preprocessing and Phase 2
# accuracy-based GridSearchCV. It intentionally does not use the Phase 2b
# cost-sensitive scorer.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

RANDOM_STATE = 42
DATASET_PATHS = {
    "Original dataset": "cancer_symptom_dataset.csv",
    "v2 dataset": "cancer_symptom_dataset_v2.csv",
}

# This is the original Phase 2 search space, with no cost-sensitive scorer.
PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8, 10, None],
    "min_samples_leaf": [1, 2, 5, 10, 15, 20],
    "criterion": ["gini", "entropy"],
}


def run_pipeline(dataset_path: str) -> dict:
    """Run the same preprocessing, split, tuning, and evaluation on one CSV."""
    df = pd.read_csv(dataset_path)
    df_clean = df.drop(columns=["patient_id"])
    subpattern = None
    if "subpattern" in df_clean.columns:
        # This is an analysis-only label. Preserve it for subgroup analysis,
        # but explicitly remove it before constructing the model feature set.
        subpattern = df_clean["subpattern"].copy()
        df_clean = df_clean.drop(columns=["subpattern"])
    df_clean["sex_encoded"] = df_clean["sex"].map({"M": 0, "F": 1})

    symptom_cols = [
        col
        for col in df_clean.columns
        if col not in ["age", "sex", "diagnosis", "sex_encoded"]
    ]
    feature_cols = symptom_cols + ["age", "sex_encoded"]
    X = df_clean[feature_cols]
    y = df_clean["diagnosis"]

    if subpattern is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        subpattern_test = None
    else:
        X_train, X_test, y_train, y_test, _, subpattern_test = train_test_split(
            X,
            y,
            subpattern,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    classes = model.classes_

    return {
        "best_params": grid_search.best_params_,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "y_test": y_test,
        "y_test_pred": pd.Series(y_test_pred, index=y_test.index),
        "subpattern_test": subpattern_test,
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, y_test_pred, labels=classes),
            index=classes,
            columns=classes,
        ),
        "classes": classes,
    }


def print_comparison(results: dict) -> None:
    original = results["Original dataset"]
    v2 = results["v2 dataset"]

    overall_table = pd.DataFrame(
        {
            "Original dataset": {
                "Test accuracy": original["test_accuracy"],
                "Train accuracy": original["train_accuracy"],
                "Train/test gap (percentage points)": (
                    original["train_accuracy"] - original["test_accuracy"]
                )
                * 100,
            },
            "v2 dataset": {
                "Test accuracy": v2["test_accuracy"],
                "Train accuracy": v2["train_accuracy"],
                "Train/test gap (percentage points)": (
                    v2["train_accuracy"] - v2["test_accuracy"]
                )
                * 100,
            },
        }
    )
    print("\n=== Overall Accuracy Comparison ===")
    print(overall_table.to_string(float_format=lambda value: f"{value:.4f}"))

    metric_rows = []
    metric_names = ["precision", "recall", "f1-score"]
    for class_name in original["classes"]:
        for metric_name in metric_names:
            metric_rows.append(
                {
                    "Class": class_name,
                    "Metric": metric_name,
                    "Original dataset": original["classification_report"][class_name][
                        metric_name
                    ],
                    "v2 dataset": v2["classification_report"][class_name][
                        metric_name
                    ],
                }
            )

    print("\n=== Per-Class Precision / Recall / F1 Comparison ===")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    confusion_matrices = pd.concat(
        {
            "Original dataset": original["confusion_matrix"],
            "v2 dataset": v2["confusion_matrix"],
        },
        axis=1,
    )
    print("\n=== Confusion Matrix Comparison ===")
    print("Rows are actual classes; columns are predicted classes.")
    print(confusion_matrices)

    # The comparison tolerance is a reporting convention for calling a metric
    # "roughly unchanged," not part of model training or selection.
    roughly_unchanged_tolerance = 0.05
    class_metrics = {
        class_name: {
            metric_name: (
                original["classification_report"][class_name][metric_name],
                v2["classification_report"][class_name][metric_name],
            )
            for metric_name in ["precision", "recall"]
        }
        for class_name in original["classes"]
    }

    def metric_change(class_name: str, metric_name: str) -> float:
        original_value, v2_value = class_metrics[class_name][metric_name]
        return v2_value - original_value

    print("\n=== Plain-Language Ablation Interpretation ===")
    for class_name in ["Breast Cancer", "Cervical Cancer"]:
        precision_change = metric_change(class_name, "precision")
        recall_change = metric_change(class_name, "recall")
        precision_direction = "dropped" if precision_change < 0 else "did not drop"
        recall_direction = "dropped" if recall_change < 0 else "did not drop"
        print(
            f"{class_name}: precision {precision_direction} "
            f"({precision_change:+.4f}); recall {recall_direction} "
            f"({recall_change:+.4f})."
        )

    lung_precision_change = metric_change("Lung Cancer", "precision")
    lung_recall_change = metric_change("Lung Cancer", "recall")
    lung_unchanged = (
        abs(lung_precision_change) <= roughly_unchanged_tolerance
        and abs(lung_recall_change) <= roughly_unchanged_tolerance
    )
    lung_summary = "stayed roughly unchanged" if lung_unchanged else "changed materially"
    print(
        f"Lung Cancer {lung_summary}: precision change "
        f"{lung_precision_change:+.4f}; recall change "
        f"{lung_recall_change:+.4f}. Its profile was not modified."
    )
    print(
        "This does not make the dataset uniformly harder: it specifically tests "
        "robustness where the original model could rely on a single near-perfect "
        "signature symptom, while leaving the Lung Cancer profile untouched."
    )


def print_v2_subpattern_breakdown(v2_result: dict) -> None:
    """Measure No Red Flags errors by the analysis-only v2 sub-pattern label."""
    if v2_result["subpattern_test"] is None:
        raise ValueError("The v2 comparison requires the 'subpattern' column.")

    test_analysis = pd.DataFrame(
        {
            "true_diagnosis": v2_result["y_test"],
            "predicted_diagnosis": v2_result["y_test_pred"],
            "subpattern": v2_result["subpattern_test"],
        }
    )
    no_red_flags_test = test_analysis[
        test_analysis["true_diagnosis"] == "No Red Flags (Routine/Benign)"
    ]

    subpattern_names = ["generic_mild", "breast_lookalike", "gyn_lookalike"]
    breakdown_rows = []
    for subpattern_name in subpattern_names:
        subpattern_test = no_red_flags_test[
            no_red_flags_test["subpattern"] == subpattern_name
        ]
        misclassified = subpattern_test[
            subpattern_test["predicted_diagnosis"]
            != "No Red Flags (Routine/Benign)"
        ]
        misclassified_as_counts = misclassified["predicted_diagnosis"].value_counts().sort_index()
        misclassified_as = ", ".join(
            f"{class_name} ({count})"
            for class_name, count in misclassified_as_counts.items()
        )

        breakdown_rows.append(
            {
                "sub-pattern": subpattern_name,
                "n": len(subpattern_test),
                "correct": len(subpattern_test) - len(misclassified),
                "misclassified": len(misclassified),
                "misclassified-as": misclassified_as or "None",
            }
        )

    breakdown_table = pd.DataFrame(breakdown_rows)
    print("\n=== v2 No Red Flags Sub-Pattern Breakdown ===")
    print(f"No Red Flags patients in the test set: {len(no_red_flags_test)}")
    print(breakdown_table.to_string(index=False))

    generic_row = breakdown_table[
        breakdown_table["sub-pattern"] == "generic_mild"
    ].iloc[0]
    lookalike_rows = breakdown_table[
        breakdown_table["sub-pattern"].isin(
            ["breast_lookalike", "gyn_lookalike"]
        )
    ]
    generic_rate = generic_row["misclassified"] / generic_row["n"]
    lookalike_n = lookalike_rows["n"].sum()
    lookalike_misclassified = lookalike_rows["misclassified"].sum()
    lookalike_rate = lookalike_misclassified / lookalike_n
    rate_difference = lookalike_rate - generic_rate

    # Treat differences within five percentage points as similar so the
    # interpretation does not overstate a small subgroup fluctuation.
    rate_similarity_tolerance = 0.05
    if abs(rate_difference) <= rate_similarity_tolerance:
        rate_comparison = "similar to"
        conclusion = "The look-alike groups do not show a meaningfully higher rate."
    elif rate_difference > 0:
        rate_comparison = "higher than"
        conclusion = "The look-alike groups have the higher rate, as predicted."
    else:
        rate_comparison = "lower than"
        conclusion = "The look-alike groups do not have the higher rate predicted."

    print("\n=== Plain-Language Sub-Pattern Interpretation ===")
    print(
        f"generic_mild misclassification rate: "
        f"{generic_row['misclassified']}/{generic_row['n']} "
        f"({generic_rate:.2%}); combined look-alike rate: "
        f"{lookalike_misclassified}/{lookalike_n} ({lookalike_rate:.2%})."
    )
    print(
        f"The combined look-alike rate is {rate_comparison} the generic_mild "
        f"rate (difference {rate_difference:+.2%}). {conclusion}"
    )


if __name__ == "__main__":
    results = {
        dataset_name: run_pipeline(dataset_path)
        for dataset_name, dataset_path in DATASET_PATHS.items()
    }

    print("\n=== Best Parameters ===")
    for dataset_name, result in results.items():
        print(f"{dataset_name}: {result['best_params']}")

    print_comparison(results)
    print_v2_subpattern_breakdown(results["v2 dataset"])
