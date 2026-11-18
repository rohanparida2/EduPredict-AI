import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from expert_system.engine import RuleEngine
from ml.explain import global_feature_importance, local_explanation
from ml.predict import FEATURE_INPUTS, predict_student
DATABASE_PATH = BASE_DIR / "database" / "app.db"
MODEL_METRICS_PATH = BASE_DIR / "ml" / "models" / "metrics.json"
GA_FEATURES_PATH = BASE_DIR / "ml" / "models" / "ga_selected_features.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "frontend" / "templates"), static_folder=str(BASE_DIR / "frontend" / "static"))
app.config["SECRET_KEY"] = os.environ.get("EDUPREDICT_SECRET_KEY", "edupredict-ai-local-development-key")
app.config["SESSION_COOKIE_HTTPONLY"] = True

FEATURE_KEYS = FEATURE_INPUTS

def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            student_id TEXT,
            department TEXT,
            semester INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            academic_data TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            payload TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()

    ensure_demo_users()


def ensure_demo_users():
    conn = get_db()
    demo_email = "demo@edupredict.ai"
    admin_email = "admin@edupredict.ai"
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (demo_email,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Demo Student", demo_email, generate_password_hash("demo123"), "student"),
        )
    existing_admin = conn.execute("SELECT id FROM users WHERE email = ?", (admin_email,)).fetchone()
    if existing_admin is None:
        conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Admin User", admin_email, generate_password_hash("admin123"), "admin"),
        )
    conn.commit()
    conn.close()


def login_required(view):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "danger")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    wrapper.__name__ = view.__name__
    return wrapper


def require_admin(view):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        user = get_user_by_id(session["user_id"])
        if user["role"] != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    wrapper.__name__ = view.__name__
    return wrapper


def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_profile(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM profiles WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_prediction(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_submission(user_id, academic_data):
    conn = get_db()
    conn.execute("INSERT INTO submissions (user_id, academic_data) VALUES (?, ?)", (user_id, json.dumps(academic_data)))
    conn.commit()
    conn.close()


def save_prediction(user_id, payload, result):
    conn = get_db()
    conn.execute("INSERT INTO predictions (user_id, payload, result) VALUES (?, ?, ?)", (user_id, json.dumps(payload), json.dumps(result)))
    conn.commit()
    conn.close()


def get_recent_submissions(user_id, limit=5):
    conn = get_db()
    rows = conn.execute("SELECT * FROM submissions WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def parse_form_data(form):
    data = {key: form.get(key, "") for key in FEATURE_KEYS}
    for key in ["semester", "backlogs", "study_consistency", "extracurricular"]:
        if data.get(key) == "":
            data[key] = 0
    for key in [
        "previous_cgpa",
        "previous_percentage",
        "internal_marks",
        "assignment_score",
        "assignment_completion_pct",
        "attendance_pct",
        "study_hours_daily",
        "mathematics_marks",
        "programming_marks",
        "core_theory_marks",
        "lab_practical_marks",
        "communication_marks",
    ]:
        try:
            data[key] = float(data.get(key, 0))
        except ValueError:
            data[key] = 0.0
    try:
        data["semester"] = int(data.get("semester", 0))
        data["backlogs"] = int(data.get("backlogs", 0))
        data["study_consistency"] = int(data.get("study_consistency", 0))
        data["extracurricular"] = int(data.get("extracurricular", 0))
    except ValueError:
        data["semester"] = 0
        data["backlogs"] = 0
        data["study_consistency"] = 0
        data["extracurricular"] = 0
    return data


def is_valid_student_data(data):
    checks = {
        "department": data.get("department") in {"CSE", "ECE", "Mechanical", "Civil", "IT"},
        "semester": 2 <= int(data.get("semester", 0)) <= 8,
        "previous_cgpa": 0 <= float(data.get("previous_cgpa", -1)) <= 10,
        "previous_percentage": 0 <= float(data.get("previous_percentage", -1)) <= 100,
        "internal_marks": 0 <= float(data.get("internal_marks", -1)) <= 50,
        "assignment_score": 0 <= float(data.get("assignment_score", -1)) <= 20,
        "assignment_completion_pct": 0 <= float(data.get("assignment_completion_pct", -1)) <= 100,
        "attendance_pct": 0 <= float(data.get("attendance_pct", -1)) <= 100,
        "study_hours_daily": 0 <= float(data.get("study_hours_daily", -1)) <= 16,
        "backlogs": 0 <= int(data.get("backlogs", -1)) <= 6,
        "study_consistency": 1 <= int(data.get("study_consistency", 0)) <= 5,
        "extracurricular": int(data.get("extracurricular", -1)) in {0, 1},
        "mathematics_marks": 0 <= float(data.get("mathematics_marks", -1)) <= 100,
        "programming_marks": 0 <= float(data.get("programming_marks", -1)) <= 100,
        "core_theory_marks": 0 <= float(data.get("core_theory_marks", -1)) <= 100,
        "lab_practical_marks": 0 <= float(data.get("lab_practical_marks", -1)) <= 100,
        "communication_marks": 0 <= float(data.get("communication_marks", -1)) <= 100,
    }
    return all(checks.values())


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Name, email, and a password of at least 6 characters are required.", "danger")
            return render_template("register.html")
        conn = get_db()
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            flash("An account with that email already exists.", "warning")
            return render_template("register.html")
        conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'student')",
            (name, email, generate_password_hash(password)),
        )
        conn.commit()
        user_id = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()[0]
        conn.close()
        session["user_id"] = user_id
        flash("Registration successful.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = get_db()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            session["role"] = row["role"]
            flash("Welcome back.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = get_user_by_id(session["user_id"])
    profile = get_profile(session["user_id"])
    prediction = get_latest_prediction(session["user_id"])
    if prediction:
        result = json.loads(prediction["result"])
    else:
        result = None
    return render_template("dashboard.html", user=user, profile=profile, prediction=result)


@app.route("/student-profile", methods=["GET", "POST"])
@login_required
def student_profile():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        student_id = request.form.get("student_id", "").strip()
        department = request.form.get("department", "")
        semester = request.form.get("semester", "")
        try:
            semester = int(request.form.get("semester", ""))
        except (TypeError, ValueError):
            semester = 0
        if (
            not name
            or not student_id
            or department not in {"CSE", "ECE", "Mechanical", "Civil", "IT"}
            or not 2 <= semester <= 8
        ):
            flash("Profile details are incomplete or invalid.", "danger")
            return redirect(url_for("student_profile"))
        conn = get_db()
        conn.execute(
            "INSERT INTO profiles (user_id, student_id, department, semester) VALUES (?, ?, ?, ?)",
            (session["user_id"], student_id, department, semester),
        )
        conn.commit()
        conn.close()
        conn = get_db(); conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, session["user_id"])); conn.commit(); conn.close()
        flash("Profile updated.", "success")
        return redirect(url_for("dashboard"))
    profile = get_profile(session["user_id"])
    user = get_user_by_id(session["user_id"])
    return render_template("profile.html", user=user, profile=profile)


@app.route("/academic-details", methods=["GET", "POST"])
@login_required
def academic_details():
    if request.method == "POST":
        data = parse_form_data(request.form)
        if not is_valid_student_data(data):
            flash("Academic data is invalid. Please check the values and ranges.", "danger")
            return render_template("academic_details.html")
        save_submission(session["user_id"], data)
        flash("Academic data saved successfully.", "success")
        return redirect(url_for("predict_route"))
    return render_template("academic_details.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_route():
    if request.method == "POST":
        data = parse_form_data(request.form)
        if not is_valid_student_data(data):
            flash("Please fix validation errors before predicting.", "danger")
            return render_template("prediction.html", data=data)
        try:
            result = predict_student(data)
        except FileNotFoundError:
            flash("Model artifacts are missing. Run the training script first.", "danger")
            return render_template("prediction.html", data=data)
        engine = RuleEngine()
        score = {
            "predicted_cgpa": result["predicted_cgpa"],
            "performance_category": result["performance_category"],
            "pass_probability": result["pass_probability"],
        }
        expert = engine.evaluate(data, score)
        save_prediction(session["user_id"], data, {**result, "expert": expert})
        return render_template("prediction.html", data=data, result=result, expert=expert)
    profile = get_profile(session["user_id"])
    data = {"department": profile["department"] if profile else "CSE", "semester": profile["semester"] if profile else 5}
    return render_template("prediction.html", data=data)


@app.route("/analytics")
@login_required
def analytics():
    prediction = get_latest_prediction(session["user_id"])
    result = json.loads(prediction["result"]) if prediction else None
    feature_importance = global_feature_importance()
    return render_template("analytics.html", result=result, feature_importance=feature_importance)


@app.route("/ai-lab")
@login_required
def ai_lab():
    metrics = {}
    ga = {}
    if MODEL_METRICS_PATH.exists():
        metrics = json.loads(MODEL_METRICS_PATH.read_text())
    if GA_FEATURES_PATH.exists():
        ga = json.loads(GA_FEATURES_PATH.read_text())
    return render_template("ai_lab.html", metrics=metrics, ga=ga, feature_importance=global_feature_importance())


@app.route("/what-if", methods=["GET", "POST"])
@login_required
def what_if():
    baseline = {
        "attendance_pct": 80,
        "study_hours_daily": 4.0,
        "assignment_completion_pct": 75,
        "internal_marks": 30,
        "department": "CSE",
        "semester": 5,
        "previous_cgpa": 7.1,
        "previous_percentage": 70,
        "assignment_score": 14,
        "backlogs": 1,
        "study_consistency": 3,
        "extracurricular": 0,
        "mathematics_marks": 60,
        "programming_marks": 65,
        "core_theory_marks": 58,
        "lab_practical_marks": 66,
        "communication_marks": 70,
    }
    if request.method == "POST":
        data = {**baseline, **parse_form_data(request.form)}
        original = predict_student(baseline)
        new = predict_student(data)
        difference = round(new["predicted_cgpa"] - original["predicted_cgpa"], 2)
        return render_template("what_if.html", baseline=baseline, data=data, original=original, new=new, difference=difference)
    return render_template("what_if.html", baseline=baseline)


@app.route("/reports")
@login_required
def reports():
    prediction = get_latest_prediction(session["user_id"])
    result = json.loads(prediction["result"]) if prediction else None
    if result:
        explanation = local_explanation(json.loads(prediction["payload"]))
    else:
        explanation = []
    return render_template("reports.html", result=result, explanation=explanation)


@app.route("/download-report")
@login_required
def download_report():
    prediction = get_latest_prediction(session["user_id"])
    if not prediction:
        flash("Make a prediction before generating the PDF report.", "warning")
        return redirect(url_for("reports"))
    payload = json.loads(prediction["payload"])
    result = json.loads(prediction["result"])
    engine = RuleEngine()
    expert = engine.evaluate(payload, {"predicted_cgpa": result["predicted_cgpa"], "performance_category": result["performance_category"], "pass_probability": result["pass_probability"]})
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdf_path = BASE_DIR / "reports" / "student_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("EduPredict AI – Student Performance Report", styles["Title"]))
    story.append(Paragraph("Synthetic dataset note: results are estimates based on synthetic educational data and are not guarantees.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    student = [
        ["Student Input", "Value"],
        *[[key, str(value)] for key, value in payload.items() if key in FEATURE_KEYS],
    ]
    story.append(Table(student, colWidths=[220, 200]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Predicted CGPA: {result['predicted_cgpa']}", styles["BodyText"]))
    story.append(Paragraph(f"Performance Category: {result['performance_category']}", styles["BodyText"]))
    story.append(Paragraph(f"Pass Probability: {result['pass_probability']}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    rules = [Paragraph(r, styles["BodyText"]) for r in expert["recommendations"]]
    story.extend(rules)
    doc.build(story)
    return send_file(str(pdf_path), as_attachment=True, download_name="student_report.pdf")


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


@app.route("/admin")
@login_required
@require_admin
def admin_dashboard():
    conn = get_db()
    total_students = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'student'").fetchone()[0]
    conn.close()
    return render_template("admin.html", total_students=total_students)


init_db()


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
