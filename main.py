import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data
data = {
    "hours": [1, 2, 3, 4, 5, 6],
    "attendance": [40, 50, 60, 70, 80, 90],
    "result": [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["hours", "attendance"]]
y = df["result"]

model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")