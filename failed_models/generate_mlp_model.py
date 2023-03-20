# using multi label Multilayer precepton to classify the ECG data

from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np


#  X_train_biased, Y_train_biased, are saved in the root folder as .npy files
#  load them using np.load()
X_train_biased = np.load('X_train_biased.npy')
Y_train_biased = np.load('Y_train_biased.npy')

# train the model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                    verbose=10)

mlp.fit(X_train_biased, Y_train_biased)

# save the  MLP model
dump(mlp, 'mlp_biased.joblib')
