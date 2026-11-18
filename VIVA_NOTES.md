# Viva Notes

## 1. Why this project is hybrid AI

This system combines:

- Machine learning for numerical and categorical prediction.
- A rule-based expert system for interpretable recommendations.
- Explainable AI to show why the prediction changed.
- A genetic algorithm for feature selection and improvement analysis.

## 2. Synthetic dataset limitation

The dataset is fully synthetic and was generated from hidden traits and random noise, not from real student records. This means the model can achieve high apparent accuracy because the relationships in the data were created programmatically rather than observed from a real academic population. The app labels this clearly in UI, reports, and documentation.

## 3. Class imbalance and leakage prevention

The dataset is imbalanced across performance categories and pass/fail. The training pipeline uses `class_weight='balanced'` for the logistic model and stratified splits to preserve class ratios. The feature set intentionally excludes `student_id`, `final_cgpa`, `performance_category`, and `pass_fail` from `X`, preventing target leakage.

## 4. Model explanations

The regression model uses Random Forest feature importance and SHAP TreeExplainer values. The XAI layer explains which inputs increased or decreased the predicted CGPA. This gives a human-readable narrative that connects to academic reasoning, not just raw coefficients.

## 5. Expert system reasoning

The expert system reads a student's inputs plus model outputs and fires rules based on threshold conditions like low attendance, poor study hours, or a backlog-heavy profile. The engine resolves conflicts by rule priority and returns ranked recommendations along with a study-plan summary.

## 6. Genetic algorithm

The GA feature-selection module evaluates subsets of encoded features using a cross-validated Random Forest objective with a small penalty for including more features. It will suggest a reduced feature set when it does not materially hurt performance, otherwise the project reports the comparison honestly.

## 7. What-if analysis

What-if analysis uses the trained models to estimate the effect of realistic changes in attendance, study hours, assignment completion, and internal marks. Results are estimates, not guarantees, and are clearly labeled as such in the UI.
