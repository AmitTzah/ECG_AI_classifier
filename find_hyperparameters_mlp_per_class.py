from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np


# Load the data
X_train = np.load('train_val_numpy_arrays/X_train_balanced_abnormal QRS.npy')
Y_train = np.load('train_val_numpy_arrays/Y_train_balanced_abnormal QRS.npy')

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100, 50, 50), (100, 100, 100), (100, 50, 50, 50, 50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'learning_rate_init': [0.01, 0.001, 0.0001],
    'early_stopping': [True, False],
    'tol': [1e-4, 1e-5, 1e-6],

}

# Create the MLPClassifier
clf = MLPClassifier(max_iter=100, verbose=10)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=4, verbose=10)

grid_search.fit(X_train, Y_train)

print("best hyperparameters: ", grid_search.best_params_)

# save the best hyperparameters to a text file
with open('best_hyperparams_mlp.txt', 'w') as f:

    f.write(str(grid_search.best_params_))
