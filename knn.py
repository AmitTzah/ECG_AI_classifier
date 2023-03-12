# using KNN to classify the ECG data

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import numpy as np

#  Y_test, Y_train, X_test, X_train are saved in the root folder as .npy files
#  load them using np.load()
Y_test = np.load('Y_test.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
X_train = np.load('X_train.npy')


# find the best K and grpah the K vs accuracy
# keep all the fitted models in a list
# use GridSearchCV
# define the parameter values that should be searched
k_range = list(range(1, 20))
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
# instantiate the grid, activate the progress bar
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, verbose=10)
# fit the grid with data
grid.fit(X_train, Y_train)
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
# plot the results
grid_mean_scores = grid.cv_results_['mean_test_score']
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# save the best KNN model
dump(grid.best_estimator_, 'best_knn.joblib')
