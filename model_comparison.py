import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# =========================
# Create output folder
os.makedirs("static", exist_ok=True)

# =========================
# Load datasets
accidents = pd.read_csv("Accidents.csv", encoding="latin1")
vehicles = pd.read_csv("Vehicles.csv", encoding="latin1")

df = accidents.merge(vehicles, on="Accident_Index")

print("Original dataset size:", len(df))

# =========================
# SPEED OPTIMIZATION: SAMPLE DATA
if len(df) > 300000:
    df = df.sample(300000, random_state=42)

print("Dataset size used for training:", len(df))

# =========================
# Target variable
target = "Accident_Severity"

label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

y = df[target]

# =========================
# Feature selection
features = [
    "Day_of_Week",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area",
    "Speed_limit",
    "Road_Type",
    "Junction_Control",
    "Vehicle_Type",
    "Age_of_Driver",
    "Sex_of_Driver",
    "Age_of_Vehicle",
    "Engine_Capacity_(CC)"
]

# Add important features if available
if "Number_of_Vehicles" in df.columns:
    features.append("Number_of_Vehicles")

if "Number_of_Casualties" in df.columns:
    features.append("Number_of_Casualties")

features = [f for f in features if f in df.columns]

print("Using features:", features)

X = df[features].copy()

# =========================
# Encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# =========================
# Handle missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X = X.astype(float)

# =========================
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Handle class imbalance (FASTER THAN SMOTEENN)
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Balanced dataset size:", X_train_res.shape)

# =========================
# Define models (FASTER SETTINGS)
models = {

    "Logistic_Regression": LogisticRegression(
        max_iter=800,
        class_weight="balanced"
    ),

    "Random_Forest": RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,

        n_estimators=220,
        max_depth=6,
        learning_rate=0.07,

        subsample=0.8,
        colsample_bytree=0.8,

        tree_method="hist",
        n_jobs=-1
    )
}

results = {}
trained_models = {}

# =========================
# Train models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "accuracy": float(acc),
        "weighted_f1": float(f1)
    }

    trained_models[name] = model

    print("Accuracy:", acc)
    print("Weighted F1:", f1)

    print(classification_report(y_test, y_pred))

    # =========================
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(name + " Confusion Matrix")
    plt.tight_layout()

    plt.savefig(f"static/{name}_cm.png")
    plt.close()

# =========================
# Save results
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)

# =========================
# Accuracy comparison chart
plt.figure(figsize=(6,4))

plt.bar(
    list(results.keys()),
    [v["accuracy"] for v in results.values()]
)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("static/model_accuracy.png")
plt.close()

# =========================
# Save best model
best_model_name = max(results, key=lambda x: results[x]["weighted_f1"])

print("\nBest Model:", best_model_name)

best_model = trained_models[best_model_name]

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("\nTraining completed successfully")