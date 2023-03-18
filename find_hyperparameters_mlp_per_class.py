from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np


# Load the data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100), (50, 50, 50)],

}

# Create the MLPClassifier
clf = MLPClassifier(max_iter=35, verbose=10)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=None, verbose=10)

y = Y_train[:, 3]

grid_search.fit(X_train, y)

print("best hyperparameters: ", grid_search.best_params_)

# save the best hyperparameters to a text file
with open('best_hyperparams_mlp.txt', 'w') as f:

    f.write(str(grid_search.best_params_))
