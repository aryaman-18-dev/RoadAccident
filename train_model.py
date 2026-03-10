import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop Time column
if "Time" in df.columns:
    df = df.drop("Time", axis=1)

# 🔥 Select ONLY features used in frontend
selected_features = [
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Weather_conditions",
    "Road_surface_conditions",
    "Light_conditions",
    "Type_of_collision"
]

X = df[selected_features]
y = df["Accident_severity"]

# Preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, selected_features)
])

# Model Pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model_pipeline, open("model.pkl", "wb"))

print("Model retrained & saved successfully!")