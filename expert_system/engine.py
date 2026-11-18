import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RULES_PATH = BASE_DIR / "expert_system" / "rules.json"


class RuleEngine:
    def __init__(self, rules_path=None):
        self.rules_path = rules_path or RULES_PATH
        self.rules = self._load_rules()

    def _load_rules(self):
        data = json.loads(self.rules_path.read_text(encoding="utf-8"))
        return sorted(data, key=lambda rule: rule.get("priority", 0), reverse=True)

    def _parse_condition(self, raw_condition):
        raw = str(raw_condition).strip()
        for op in ["<=", ">=", "<>", "=", "<", ">"]:
            if op in raw:
                return op, raw.split(op, 1)[1].strip()
        return "==", raw

    def _matches(self, rule, context):
        matched = []
        for key, raw_value in rule.get("conditions", {}).items():
            value = context.get(key)
            operator, expected_value = self._parse_condition(raw_value)
            if value is None:
                matched.append((key, False, "missing value"))
                continue
            if operator in {"==", "="}:
                ok = str(value).strip() == str(expected_value).strip()
                matched.append((key, ok, f"{key}={value} expected {expected_value}"))
            elif operator == "<":
                try:
                    ok = float(value) < float(expected_value)
                    matched.append((key, ok, f"{key}={value} is less than {expected_value}"))
                except (TypeError, ValueError):
                    matched.append((key, False, f"{key}={value} cannot compare with {expected_value}"))
            elif operator == ">":
                try:
                    ok = float(value) > float(expected_value)
                    matched.append((key, ok, f"{key}={value} is greater than {expected_value}"))
                except (TypeError, ValueError):
                    matched.append((key, False, f"{key}={value} cannot compare with {expected_value}"))
            elif operator == "<=":
                try:
                    ok = float(value) <= float(expected_value)
                    matched.append((key, ok, f"{key}={value} is at most {expected_value}"))
                except (TypeError, ValueError):
                    matched.append((key, False, f"{key}={value} cannot compare with {expected_value}"))
            elif operator == ">=":
                try:
                    ok = float(value) >= float(expected_value)
                    matched.append((key, ok, f"{key}={value} is at least {expected_value}"))
                except (TypeError, ValueError):
                    matched.append((key, False, f"{key}={value} cannot compare with {expected_value}"))
            elif operator == "<>":
                ok = str(value).strip() != str(expected_value).strip()
                matched.append((key, ok, f"{key}={value} is not {expected_value}"))
            else:
                matched.append((key, False, "unsupported operator"))
        return matched

    def evaluate(self, student_data, ml_output):
        context = dict(student_data)
        context.update(ml_output)
        fired = []
        trace = []
        for rule in self.rules:
            matches = self._matches(rule, context)
            all_match = all(status for _, status, _ in matches)
            if all_match:
                fired.append(rule)
                trace.append({
                    "id": rule["id"],
                    "priority": rule.get("priority", 0),
                    "recommendation": rule["recommendation"],
                    "reasons": [reason for _, _, reason in matches],
                })
        recommendations = [r["recommendation"] for r in fired]
        return {
            "rules_fired": fired,
            "recommendations": recommendations,
            "rule_trace": trace,
            "weekly_plan": self.generate_weekly_plan(student_data, ml_output, fired),
            "weak_subjects": self.weak_subjects(student_data),
        }

    def weak_subjects(self, student_data):
        marks = {
            "mathematics": float(student_data.get("mathematics_marks", 0)),
            "programming": float(student_data.get("programming_marks", 0)),
            "core_theory": float(student_data.get("core_theory_marks", 0)),
            "lab_practical": float(student_data.get("lab_practical_marks", 0)),
            "communication": float(student_data.get("communication_marks", 0)),
        }
        return sorted(marks.items(), key=lambda item: item[1])[:3]

    def generate_weekly_plan(self, student_data, ml_output, fired_rules):
        weak_subjects = self.weak_subjects(student_data)
        plan = []
        if weak_subjects:
            plan.append(f"Day 1-2: Revise {weak_subjects[0][0]} fundamentals for 90 minutes each day.")
            plan.append(f"Day 3-4: Practice {weak_subjects[1][0]} with mini quizzes and problem sets.")
            plan.append(f"Day 5: Review {weak_subjects[2][0]} and update notes with formulas or code snippets.")
        attendance = float(student_data.get("attendance_pct", 0))
        if attendance < 80:
            plan.append("Attend all lectures this week and catch up on missed content before the next assessment.")
        hours = float(student_data.get("study_hours_daily", 0))
        if hours < 3:
            plan.append("Add 1.5 extra hours per day to your study schedule for active revision and short tests.")
        if fired_rules:
            plan.append("Use the fired rule priorities to schedule the hardest tasks first each morning.")
        if len(plan) < 3:
            plan.append("Maintain revision consistency and complete at least one timed practice session each day.")
        return plan


if __name__ == "__main__":
    engine = RuleEngine()
    sample_student = {
        "department": "CSE",
        "semester": 5,
        "attendance_pct": 71,
        "study_hours_daily": 1.5,
        "assignment_completion_pct": 56,
        "backlogs": 2,
        "internal_marks": 21,
        "mathematics_marks": 44,
        "programming_marks": 48,
        "core_theory_marks": 52,
        "lab_practical_marks": 53,
        "communication_marks": 63,
        "study_consistency": 2,
        "extracurricular": 1,
    }
    sample_ml = {"predicted_cgpa": 5.8, "performance_category": "Average", "pass_probability": 0.49}
    print(engine.evaluate(sample_student, sample_ml))
