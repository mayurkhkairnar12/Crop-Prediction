from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open("knr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load your preprocessor too (important!)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Year = int(request.form["Year"])
    rain_fall = float(request.form["rain_fall"])
    pressure = float(request.form["pressure"])
    avg = float(request.form["avg"])
    Area = request.form["Area"]
    Item = request.form["Item"]

    features = np.array([[Year, rain_fall, avg, pressure, Area, Item]])
    transformed_features = preprocessor.transform(features)
    result = model.predict(transformed_features).reshape(1, -1)

    return render_template("index.html", prediction_text=f"Predicted {Item} yield in {Area} {Year} is {result[0][0]:.2f} hg/ha_yield")

if __name__ == "__main__":
    app.run(debug=True)
