import json
from pathlib import Path

import joblib
import pandas as pd

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


def _ensure_model_ready():
    required = [
        MODEL_DIR / "random_forest_regression.joblib",
        MODEL_DIR / "decision_tree_classifier.joblib",
        MODEL_DIR / "logistic_regression.joblib",
        MODEL_DIR / "feature_columns.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained model artifacts: {missing}")


def _build_input_frame(raw_data):
    row = {key: raw_data.get(key, 0) for key in FEATURE_INPUTS}
    cleaned = {}
    for key, value in row.items():
        if value is None or value == "":
            cleaned[key] = 0
        else:
            try:
                cleaned[key] = float(value) if key != "department" else str(value)
            except (TypeError, ValueError):
                cleaned[key] = value
    return pd.DataFrame([cleaned], columns=FEATURE_INPUTS)


def _subject_risk_lookup(data):
    marks = {
        "mathematics": float(data.get("mathematics_marks", 0)),
        "programming": float(data.get("programming_marks", 0)),
        "core_theory": float(data.get("core_theory_marks", 0)),
        "lab_practical": float(data.get("lab_practical_marks", 0)),
        "communication": float(data.get("communication_marks", 0)),
    }
    subject_risk = {}
    for subject, value in marks.items():
        risk = max(0.0, 100.0 - value)
        if value < 50:
            risk = max(risk, 40.0)
        subject_risk[subject] = round(risk / 100.0, 3)
    return subject_risk


def predict_student(raw_data):
    _ensure_model_ready()
    df = _build_input_frame(raw_data)
    reg_model = joblib.load(MODEL_DIR / "random_forest_regression.joblib")
    clf_model = joblib.load(MODEL_DIR / "decision_tree_classifier.joblib")
    log_model = joblib.load(MODEL_DIR / "logistic_regression.joblib")

    predicted_cgpa = float(reg_model.predict(df)[0])
    predicted_category = str(clf_model.predict(df)[0])
    probabilities = log_model.predict_proba(df)[0]
    positive_index = 1 if log_model.named_steps["model"].classes_[1] == 1 else 0
    pass_probability = float(probabilities[positive_index])
    confidence = float(max(probabilities))

    topic_risk = _subject_risk_lookup(raw_data)

    return {
        "predicted_cgpa": round(predicted_cgpa, 2),
        "performance_category": predicted_category,
        "pass_probability": round(pass_probability, 3),
        "pass_fail": 1 if pass_probability >= 0.5 else 0,
        "confidence": round(confidence, 3),
        "subject_risk": topic_risk,
        "feature_names": json.loads((MODEL_DIR / "feature_columns.json").read_text()),
    }


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
    print(predict_student(sample))
