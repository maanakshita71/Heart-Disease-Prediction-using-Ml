from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("pipeline.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Age": int(request.form["Age"]),
            "Sex": request.form["Sex"],
            "ChestPain": request.form["ChestPain"],
            "RestBP": int(request.form["RestBP"]),
            "Chol": int(request.form["Chol"]),
            "Fbs": int(request.form["Fbs"]),
            "RestECG": request.form["RestECG"],
            "MaxHR": int(request.form["MaxHR"]),
            "ExAng": request.form["ExAng"],
            "Oldpeak": float(request.form["Oldpeak"]),
            "Slope": request.form["Slope"],
            "Ca": int(request.form["Ca"]),
            "Thal": request.form["Thal"]
        }

        df = pd.DataFrame([data])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk (Healthy)"

        return render_template("index.html",
                               prediction_text=result,
                               probability=round(prob*100, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)