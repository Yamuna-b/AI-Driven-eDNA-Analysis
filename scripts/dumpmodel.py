import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("models/edna_model.pkl", "wb") as f:
    pickle.dump(model, f)