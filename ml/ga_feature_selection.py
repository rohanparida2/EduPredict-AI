import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "student_performance_dataset.csv"
OUTPUT_PATH = BASE_DIR / "ml" / "models" / "ga_selected_features.json"

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


def prepare_data():
    df = pd.read_csv(DATASET_PATH)
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
    X_encoded = preprocessor.fit_transform(df[FEATURE_INPUTS])
    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_encoded, df["final_cgpa"].to_numpy(), feature_names


def fitness(chromosome, X, y, feature_names):
    selected_indices = [idx for idx, flag in enumerate(chromosome) if flag]
    if not selected_indices:
        return -1.0
    selected_X = X[:, selected_indices]
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    scores = cross_val_score(model, selected_X, y, cv=3, scoring="r2")
    score = float(scores.mean())
    penalty = 0.01 * len(selected_indices)
    return score - penalty


def tournament(population, scores):
    idxs = np.random.choice(len(population), size=3, replace=False)
    best = idxs[0]
    for idx in idxs[1:]:
        if scores[idx] > scores[best]:
            best = idx
    return population[best].copy()


def crossover(parent_a, parent_b):
    point = np.random.randint(1, len(parent_a))
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def mutate(chromosome):
    mutation_index = np.random.randint(len(chromosome))
    chromosome[mutation_index] = 1 - chromosome[mutation_index]
    return chromosome


def run_ga():
    X, y, feature_names = prepare_data()
    rng = np.random.default_rng(42)
    population = np.array(
        [rng.integers(0, 2, size=len(feature_names), dtype=int) for _ in range(20)],
        dtype=int,
    )
    generation_history = []

    for generation in range(25):
        scores = np.array([fitness(chromosome, X, y, feature_names) for chromosome in population])
        generation_history.append(float(scores.max()))
        next_population = []
        while len(next_population) < len(population):
            parent_a = tournament(population, scores)
            parent_b = tournament(population, scores)
            child_a, child_b = crossover(parent_a, parent_b)
            if rng.random() < 0.15:
                child_a = mutate(child_a)
            if rng.random() < 0.15:
                child_b = mutate(child_b)
            next_population.extend([child_a, child_b])
        population = np.array(next_population[: len(population)], dtype=int)

    best = max(population, key=lambda chromosome: fitness(chromosome, X, y, feature_names))
    selected = [feature for flag, feature in zip(best, feature_names) if flag]
    result = {
        "selected_features": selected,
        "population_size": len(population),
        "generations": 25,
        "best_fitness": float(fitness(best, X, y, feature_names)),
        "generation_history": generation_history,
        "all_features_count": len(feature_names),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(run_ga())
