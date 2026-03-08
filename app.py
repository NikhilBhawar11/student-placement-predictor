from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    features = np.array([values])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Student will be PLACED 🎉"
    else:
        result = "Student will NOT be placed ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)