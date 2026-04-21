import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("heart.csv")

# Drop unwanted column
if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)

# Convert target
data["AHD"] = data["AHD"].map({"Yes": 1, "No": 0})

# Convert categorical → numeric
data = pd.get_dummies(data)

# Split features/target
X = data.drop("AHD", axis=1)
y = data["AHD"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model + columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model & columns saved!")