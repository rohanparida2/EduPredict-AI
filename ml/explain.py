import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "ml" / "models"

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


def _prepare_row(raw_data):
    row = {key: raw_data.get(key, 0) for key in FEATURE_INPUTS}
    cleaned = {}
    for key, value in row.items():
        if value is None or value == "":
            cleaned[key] = 0
        else:
            cleaned[key] = float(value) if key != "department" else str(value)
    return pd.DataFrame([cleaned], columns=FEATURE_INPUTS)


def load_model():
    return joblib.load(MODEL_DIR / "random_forest_regression.joblib")


def global_feature_importance():
    model = load_model().named_steps["model"]
    preprocessor = load_model().named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": feat, "importance": round(float(score), 4)} for feat, score in ranked[:10]]


def local_explanation(raw_data):
    model = load_model()
    input_row = _prepare_row(raw_data)
    transformed = model.named_steps["preprocessor"].transform(input_row)
    tree_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        values = shap_values[0][0]
    else:
        values = shap_values[0]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    explanation = []
    for feature_name, value in zip(feature_names, values):
        if abs(value) < 0.0001:
            continue
        direction = "raised" if value > 0 else "lowered"
        explanation.append({
            "feature": feature_name,
            "impact": round(float(value), 4),
            "summary": f"{feature_name} {direction} your prediction by {abs(round(float(value), 4)):.4f}",
        })
    explanation.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return explanation[:8]


if __name__ == "__main__":
    sample = {
        "department": "CSE",
        "semester": 5,
        "previous_cgpa": 7.2,
        "previous_percentage": 74.2,
        "internal_marks": 32.4,
        "assignment_score": 13.8,
        "assignment_completion_pct": 80.0,
        "attendance_pct": 86.0,
        "study_hours_daily": 4.5,
        "backlogs": 1,
        "study_consistency": 4,
        "extracurricular": 1,
        "mathematics_marks": 70,
        "programming_marks": 75,
        "core_theory_marks": 68,
        "lab_practical_marks": 80,
        "communication_marks": 72,
    }
    print(json.dumps(global_feature_importance()[:5], indent=2))
    print(json.dumps(local_explanation(sample), indent=2))
