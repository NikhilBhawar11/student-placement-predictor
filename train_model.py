import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("data/placementdata.csv")

# Encode categorical columns
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

data["PlacementTraining"] = le1.fit_transform(data["PlacementTraining"])
data["ExtracurricularActivities"] = le2.fit_transform(data["ExtracurricularActivities"])
data["PlacementStatus"] = le3.fit_transform(data["PlacementStatus"])

# Features and Target
X = data.drop(["PlacementStatus","StudentID"], axis=1)
y = data["PlacementStatus"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")