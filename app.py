from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

app.config["SECRET_KEY"] = "secret123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ==============================
# LOAD MODEL
# ==============================

model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


# ==============================
# DATABASE MODELS
# ==============================

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)


class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    severity = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    day = db.Column(db.String(20))
    weather = db.Column(db.String(50))


# ==============================
# LOGIN MANAGER
# ==============================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ==============================
# HOME
# ==============================

@app.route("/")
@login_required
def home():
    return render_template("index.html")


# ==============================
# REGISTER
# ==============================

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        user = User(username=username, password=password)

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# ==============================
# LOGIN
# ==============================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):

            login_user(user)
            return redirect("/")

    return render_template("login.html")


# ==============================
# LOGOUT
# ==============================

@app.route("/logout")
@login_required
def logout():

    logout_user()
    return redirect("/login")


# ==============================
# ADMIN PANEL
# ==============================

@app.route("/admin")
@login_required
def admin():

    if not current_user.is_admin:
        return "Access Denied"

    users = User.query.all()
    predictions = Prediction.query.all()

    total_users = len(users)
    total_predictions = len(predictions)

    severity_counts = {}

    for p in predictions:
        severity_counts[p.severity] = severity_counts.get(p.severity, 0) + 1

    severity_labels = list(severity_counts.keys())
    severity_values = list(severity_counts.values())

    return render_template(
        "admin.html",
        users=users,
        predictions=predictions,
        total_users=total_users,
        total_predictions=total_predictions,
        severity_labels=severity_labels,
        severity_values=severity_values
    )


# ==============================
# HISTORY
# ==============================

@app.route("/history")
@login_required
def history():

    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    return render_template("history.html", predictions=predictions)


# ==============================
# CLEAR HISTORY
# ==============================

@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():

    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()

    return redirect(url_for("history"))


# ==============================
# USER ANALYTICS
# ==============================

@app.route("/my_analytics")
@login_required
def my_analytics():

    user_preds = Prediction.query.filter_by(user_id=current_user.id).all()

    risks = [p.severity for p in user_preds] if user_preds else []

    return render_template("user_analytics.html", risks=risks)


# ==============================
# GLOBAL ANALYTICS
# ==============================

@app.route("/analytics")
def analytics():

    predictions = Prediction.query.all()
    risks = [p.severity for p in predictions]

    return render_template("analytics.html", risks=risks)


# ==============================
# PREDICT
# ==============================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    try:

        input_data = {
            "Day_of_Week": int(request.form.get("Day_of_Week")),
            "Light_Conditions": int(request.form.get("Light_Conditions")),
            "Weather_Conditions": int(request.form.get("Weather_Conditions")),
            "Road_Surface_Conditions": int(request.form.get("Road_Surface_Conditions")),
            "Urban_or_Rural_Area": int(request.form.get("Urban_or_Rural_Area")),
            "Speed_limit": int(request.form.get("Speed_limit")),
            "Road_Type": int(request.form.get("Road_Type")),
            "Junction_Control": int(request.form.get("Junction_Control")),
            "Vehicle_Type": int(request.form.get("Vehicle_Type")),
            "Sex_of_Driver": int(request.form.get("Sex_of_Driver")),
            "Age_of_Vehicle": int(request.form.get("Age_of_Vehicle")),
            "Number_of_Vehicles": int(request.form.get("Number_of_Vehicles")),
            "Number_of_Casualties": int(request.form.get("Number_of_Casualties"))
        }

        df = pd.DataFrame([input_data])

        pred_encoded = int(model.predict(df)[0])
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        probs = model.predict_proba(df)[0]
        classes = label_encoder.classes_

        probabilities = {}

        for i, c in enumerate(classes):
            probabilities[c] = round(float(probs[i]) * 100, 2)

        confidence = round(max(probabilities.values()), 2)

        new_prediction = Prediction(
            user_id=current_user.id,
            severity=prediction,
            confidence=confidence,
            day=str(input_data["Day_of_Week"]),
            weather=str(input_data["Weather_Conditions"])
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template(
            "index.html",
            prediction_text=prediction.capitalize(),
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "index.html",
            error="⚠ Prediction failed. Check inputs."
        )


# ==============================
# DASHBOARD
# ==============================

@app.route("/dashboard")
@login_required
def dashboard():

    results = {}

    if os.path.exists("model_results.json"):
        with open("model_results.json") as f:
            results = json.load(f)

    best_model = "N/A"

    if results:
        best_model = max(results, key=lambda x: results[x]["weighted_f1"])

    feature_plot_available = False

    try:

        if hasattr(model, "feature_importances_"):

            feature_names = [
                "Day_of_Week","Light_Conditions","Weather_Conditions",
                "Road_Surface_Conditions","Urban_or_Rural_Area",
                "Speed_limit","Road_Type","Junction_Control",
                "Vehicle_Type","Sex_of_Driver",
                "Age_of_Vehicle","Number_of_Vehicles","Number_of_Casualties"
            ]

            importances = model.feature_importances_

            indices = np.argsort(importances)[-10:]

            plt.figure(figsize=(8,6))

            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), np.array(feature_names)[indices])

            plt.title("Top 10 Feature Importance")
            plt.tight_layout()

            plt.savefig("static/feature_importance.png")
            plt.close()

            feature_plot_available = True

    except Exception as e:
        print("Feature importance error:", e)

    return render_template(
        "dashboard.html",
        results=results,
        best_model=best_model,
        feature_plot_available=feature_plot_available
    )


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)