from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        hours = int(request.form["hours"])
        attendance = int(request.form["attendance"])

        result = model.predict([[hours, attendance]])[0]
        prediction = "PASS ✅" if result == 1 else "FAIL ❌"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()