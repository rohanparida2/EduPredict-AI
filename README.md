# EduPredict AI

EduPredict AI is a hybrid AI student performance prediction system built for a college mini project in Principles of Artificial Intelligence. It combines machine learning, expert-system reasoning, explainable AI, and a GA-driven feature selection workflow.

## Features

- Student auth with registration/login/logout
- Academic data form with validation
- Random Forest regression for final CGPA
- Decision Tree classification for performance category
- Logistic Regression for pass/fail probability
- Expert-system recommendations and weekly plan generation
- SHAP-based explainability
- GA feature selection and model comparison
- What-if analysis and PDF report generation
- Responsive admin/student dashboard

## Synthetic dataset notice

The project is trained only on the provided synthetic dataset in `dataset/student_performance_dataset.csv` and uses `database/student_performance.db` as a dataset reference only. This is not real student data and should be treated as educational/demo data.

## Setup

From the project folder, run `run.bat` on Windows, or use these commands:

1. Create and activate a virtual environment (Python 3.11 or newer).
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Train the models only if files under `ml/models/` are missing: `python -m ml.train`
4. Start the app: `python -m backend.app`
5. Open http://127.0.0.1:5000 in your browser.

The Flask server stays open in the terminal while the browser app is in use. Press Ctrl+C in that terminal to stop it. The app database is initialized automatically on first launch.

### Demo account

- Email: demo@edupredict.ai
- Password: demo123

## Project structure

- `dataset/` — provided synthetic data and generator
- `database/` — dataset copy plus runtime app database
- `ml/` — training, prediction, GA, explanation code and saved models
- `expert_system/` — rules and inference engine
- `backend/` — Flask app and authentication
- `frontend/` — HTML templates and static assets

## Important notes

- `app.db` is the app database for user records and prediction history.
- `student_performance.db` stays separate and is used only as a dataset source for SQL reference.
- The app intentionally reports synthetic-data caveats in the UI and PDF output.
