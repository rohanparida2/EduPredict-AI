"""
EduPredict AI - Synthetic Student Performance Dataset Generator
================================================================
NOTE: This dataset is FULLY SYNTHETIC. It is NOT real student data and model
accuracy on it must NOT be presented as validated on real-world students.

Method: each student has a hidden "ability" and "effort" trait. Observed
features are generated from these traits + random noise, and the final
CGPA is produced from the traits + features + noise (so it is NOT a clean
formula of the inputs). Relationships mirror those seen in education
research and in the UCI Student Performance dataset (e.g. attendance,
study time and previous grades correlate positively with results;
backlogs correlate negatively).

Run:  python generate_dataset.py
Out:  student_performance_dataset.csv and student_performance.db
"""
import sqlite3
import numpy as np
import pandas as pd

SEED = 42
N = 800
rng = np.random.default_rng(SEED)

clip = np.clip

# ---- hidden traits (not saved in the dataset) ----
ability = rng.normal(0, 1, N)
effort = 0.35 * ability + rng.normal(0, 0.95, N)

# ---- categorical ----
departments = ["CSE", "ECE", "Mechanical", "Civil", "IT"]
department = rng.choice(departments, N, p=[0.30, 0.20, 0.17, 0.13, 0.20])
semester = rng.integers(2, 9, N)  # semesters 2-8 (needs a previous semester)

# ---- previous academics ----
previous_cgpa = clip(6.8 + 1.1 * ability + rng.normal(0, 0.45, N), 3.0, 10.0)
previous_percentage = clip(previous_cgpa * 9.5 + rng.normal(0, 2.5, N), 30, 100)

# ---- habits ----
attendance_pct = clip(78 + 9 * effort + 3 * ability + rng.normal(0, 6, N), 35, 100)
study_hours_daily = clip(3.2 + 1.3 * effort + 0.3 * ability + rng.normal(0, 0.9, N), 0, 12)
assignment_completion_pct = clip(80 + 10 * effort + 3 * ability + rng.normal(0, 8, N), 20, 100)
study_consistency = clip(np.round(3 + 0.9 * effort + rng.normal(0, 0.7, N)), 1, 5).astype(int)
extracurricular = (rng.random(N) < clip(0.45 + 0.05 * effort, 0.1, 0.9)).astype(int)

# ---- internal / assignment scores ----
internal_marks = clip(30 + 6 * ability + 3 * effort + rng.normal(0, 4, N), 5, 50)  # out of 50
assignment_score = clip(assignment_completion_pct * 0.18 + 2 * ability + rng.normal(0, 1.8, N), 0, 20)  # out of 20

# ---- backlogs (more likely for weak, low-effort students) ----
backlog_rate = clip(np.exp(-1.1 - 0.9 * ability - 0.5 * effort), 0.02, 3.5)
backlogs = np.minimum(rng.poisson(backlog_rate), 6)

# ---- subject marks (out of 100) ----
def subject(base, diff):
    return clip(base + 12 * ability + 4 * effort + rng.normal(0, 7, N) - diff, 10, 100)

mathematics = subject(62, 4)
programming = subject(65, 0)
core_theory = subject(63, 2)
lab_practical = subject(72, -4)
communication = subject(68, -2)
# department-specific tweak (e.g. CSE/IT stronger in programming)
prog_boost = np.where(np.isin(department, ["CSE", "IT"]), 4, -3)
programming = clip(programming + prog_boost, 10, 100)

# ---- targets ----
final_cgpa = (
    0.45 * previous_cgpa
    + 0.020 * internal_marks * 2           # internal on 0-100 scale
    + 0.018 * attendance_pct
    + 0.16 * study_hours_daily
    + 0.008 * assignment_completion_pct
    + 0.10 * study_consistency
    - 0.28 * backlogs
    + 0.12 * ability                        # hidden trait -> irreducible noise
    + rng.normal(0, 0.38, N)
)
# rescale to a realistic 3-10 range
final_cgpa = (final_cgpa - final_cgpa.mean()) / final_cgpa.std() * 1.0 + 7.1
final_cgpa = clip(final_cgpa, 3.0, 10.0)

def category(c):
    if c >= 8.5:
        return "Excellent"
    if c >= 7.0:
        return "Good"
    if c >= 5.5:
        return "Average"
    return "Needs Improvement"

performance_category = [category(c) for c in final_cgpa]

subj_min = np.minimum.reduce([mathematics, programming, core_theory, lab_practical, communication])
pass_fail = ((final_cgpa >= 5.0) & (subj_min >= 30)).astype(int)  # 1 = Pass, 0 = Fail

df = pd.DataFrame({
    "student_id": [f"STU{str(i + 1).zfill(4)}" for i in range(N)],
    "department": department,
    "semester": semester,
    "previous_cgpa": previous_cgpa.round(2),
    "previous_percentage": previous_percentage.round(1),
    "internal_marks": internal_marks.round(1),
    "assignment_score": assignment_score.round(1),
    "assignment_completion_pct": assignment_completion_pct.round(1),
    "attendance_pct": attendance_pct.round(1),
    "study_hours_daily": study_hours_daily.round(1),
    "backlogs": backlogs,
    "study_consistency": study_consistency,
    "extracurricular": extracurricular,
    "mathematics_marks": mathematics.round(1),
    "programming_marks": programming.round(1),
    "core_theory_marks": core_theory.round(1),
    "lab_practical_marks": lab_practical.round(1),
    "communication_marks": communication.round(1),
    # ---- targets (keep OUT of the feature set when training) ----
    "final_cgpa": final_cgpa.round(2),
    "performance_category": performance_category,
    "pass_fail": pass_fail,
})

df.to_csv("student_performance_dataset.csv", index=False)

con = sqlite3.connect("student_performance.db")
df.to_sql("student_dataset", con, if_exists="replace", index=False)
con.close()

print(df.shape)
print(df["performance_category"].value_counts())
print("Pass rate:", round(df["pass_fail"].mean(), 3))
