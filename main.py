KAGGLE_DATASET = "nehalbirla/vehicle-dataset-from-cardekho"   # שם הדאטה-סט ב-Kaggle
FEATURE_COLUMN = "km_driven"                       # שם עמודת המאפיין (X) - מה שממנו נחזה
TARGET_COLUMN = "selling_price"                              # שם עמודת המטרה (y) - מה שרוצים לחזות

import kagglehub

# Download latest version
path = kagglehub.dataset_download(KAGGLE_DATASET)

print("Path to dataset files:", path)

import pandas as pd
import os

csv_path = os.path.join(path, "CAR DETAILS FROM CAR DEKHO.csv")
data = pd.read_csv(csv_path)

data.head()

print(data[[FEATURE_COLUMN, TARGET_COLUMN]].isnull().sum())

import numpy as np

X = data[FEATURE_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X sample:", X[:5])
print("y sample:", y[:5])

baseline_prediction = np.mean(y)
print("baseline prediction (always the same value):", baseline_prediction)

baseline_loss = np.mean(np.abs(y - baseline_prediction))
print("baseline loss:", baseline_loss)

from sklearn.linear_model import LinearRegression

X_reshaped = X.reshape(-1, 1)
print("X shape before reshape:", X.shape)
print("X shape after reshape:", X_reshaped.shape)

model = LinearRegression()
model.fit(X_reshaped, y)

print("training done")

w = model.coef_[0]
b = model.intercept_

print("w:", w)
print("b:", b)

sample_x = X[0]
print("sample_x:", sample_x)

manual_prediction = w * sample_x + b
print("manual calculation (w * x + b):", manual_prediction)

sklearn_prediction = model.predict(np.array([[sample_x]]))
print("sklearn predict:", sklearn_prediction[0])

y_hat = model.predict(X_reshaped)

model_loss = np.mean(np.abs(y - y_hat))

print("baseline loss:", baseline_loss)
print("model loss:", model_loss)

user_input = input("enter km driven by this car: ")
user_sqft = float(user_input)

predicted_price = model.predict(np.array([[user_sqft]]))[0]

print(" המחיר החזוי:", predicted_price)
