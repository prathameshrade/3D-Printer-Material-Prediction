# save_model.py (same as before)
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1, 2, 3], [10, 10, 10], [0, 0, 0], [3, 2, 1]])
y = [1, 1, 0, 0]  # Fault or No Fault

model = LogisticRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

