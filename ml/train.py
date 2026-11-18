import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "student_performance_dataset.csv"
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_INPUTS = [
    "department",
    "semester",
    "previous_cgpa",
    "previous_percentage",
    "internal_marks",
    "assignment_score",
    "assignment_completion_pct",
    "attendance_pct",
    "study_hours_daily",
    "backlogs",
    "study_consistency",
    "extracurricular",
    "mathematics_marks",
    "programming_marks",
    "core_theory_marks",
    "lab_practical_marks",
    "communication_marks",
]
TARGETS = ["final_cgpa", "performance_category", "pass_fail"]


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    required = set(FEATURE_INPUTS + TARGETS + ["student_id"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    return df


def build_preprocessor():
    numeric_features = [
        "semester",
        "previous_cgpa",
        "previous_percentage",
        "internal_marks",
        "assignment_score",
        "assignment_completion_pct",
        "attendance_pct",
        "study_hours_daily",
        "backlogs",
        "study_consistency",
        "extracurricular",
        "mathematics_marks",
        "programming_marks",
        "core_theory_marks",
        "lab_practical_marks",
        "communication_marks",
    ]
    categorical_features = ["department"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions),
        "ExplainedVariance": explained_variance_score(y_test, predictions),
    }
    return metrics


def evaluate_classification(model, X_test, y_test):
    predictions = model.predict(X_test)
    labels = sorted(y_test.unique())
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, labels=labels, average=None, zero_division=0)
    per_class = {str(label): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for label, p, r, f, s in zip(labels, precision, recall, f1, support)}
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_recall_f1": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "f1_macro": f1_score(y_test, predictions, labels=labels, average="macro"),
        "per_class": per_class,
    }


def main():
    df = load_dataset()
    X = df[FEATURE_INPUTS]
    y_reg = df["final_cgpa"]
    y_cls = df["performance_category"]
    y_bin = df["pass_fail"]

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_cls_train,
        y_cls_test,
        y_bin_train,
        y_bin_test,
    ) = train_test_split(
        X,
        y_reg,
        y_cls,
        y_bin,
        test_size=0.25,
        random_state=42,
        stratify=y_cls,
    )

    preprocessor = build_preprocessor()

    reg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )
    reg_model.fit(X_train, y_reg_train)
    reg_metrics = evaluate_regression(reg_model, X_test, y_reg_test)

    clf_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    clf_model.fit(X_train, y_cls_train)
    cls_metrics = evaluate_classification(clf_model, X_test, y_cls_test)

    log_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(random_state=42, class_weight="balanced", max_iter=4000, solver="liblinear")),
        ]
    )
    log_model.fit(X_train, y_bin_train)
    bin_metrics = evaluate_classification(log_model, X_test, y_bin_test)

    feature_names = log_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(feature_names, indent=2))

    joblib.dump(reg_model, MODEL_DIR / "random_forest_regression.joblib")
    joblib.dump(clf_model, MODEL_DIR / "decision_tree_classifier.joblib")
    joblib.dump(log_model, MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(build_preprocessor(), MODEL_DIR / "preprocessor.joblib")

    metrics_summary = {
        "regression": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in reg_metrics.items()},
        "classification": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in cls_metrics.items() if k != "precision_recall_f1"},
        "classification_report": cls_metrics["precision_recall_f1"],
        "binary_classification": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in bin_metrics.items() if k != "precision_recall_f1"},
        "binary_classification_report": bin_metrics["precision_recall_f1"],
        "dataset_info": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "synthetic": True,
            "class_distribution": {k: int(v) for k, v in df["performance_category"].value_counts().to_dict().items()},
            "pass_rate": float(df["pass_fail"].mean()),
        },
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    print("Training complete.")
    print(json.dumps({"regression": reg_metrics, "classification": cls_metrics["accuracy"], "binary_accuracy": bin_metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
