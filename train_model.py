import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("heart.csv")

# Drop unwanted column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Target column
df["AHD"] = df["AHD"].map({"Yes": 1, "No": 0})

X = df.drop("AHD", axis=1)
y = df["AHD"]

# Categorical + Numerical columns
cat_cols = ["Sex", "ChestPain", "RestECG", "ExAng", "Slope", "Thal"]
num_cols = ["Age", "RestBP", "Chol", "Fbs", "MaxHR", "Oldpeak", "Ca"]

# Pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save
pickle.dump(pipeline, open("pipeline.pkl", "wb"))

print("✅ Pipeline trained & saved!")