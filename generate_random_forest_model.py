from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load the training data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

# Train the model
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, verbose=10)
model.fit(X_train, Y_train)


# Save the trained model to disk
joblib.dump(model, 'random_forest.joblib')
