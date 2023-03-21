from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import joblib

if 'best_hyperparams_mlp.txt' not in os.listdir():
    best_hyperparams = {'hidden_layer_sizes': (100, 100)}

else:

    with open('best_hyperparams_mlp.txt', 'r') as f:
        best_hyperparams = f.read()

    # convert the string to a dictionary

    best_hyperparams = eval(best_hyperparams)

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


# Check if a "models" folder exists, if not create one
# if it does, remove all the files in it and then create it

if 'models' in os.listdir():

    for file in os.listdir('models'):

        # remove all the files in the models folder

        os.remove(os.path.join('models', file))


else:
    os.mkdir('models')

X_train_check = np.load('train_val_numpy_arrays/X_train.npy')
Y_train_check = np.load('train_val_numpy_arrays/Y_train.npy')


# Partition the problem into 7 different single label classes
for i in range(7):

    if i == 0:
        X_train = np.load(os.path.join(
            "train_val_numpy_arrays", f'X_train_{classes_names[i]}.npy'))
        Y_train = np.load(os.path.join("train_val_numpy_arrays",
                                       f'y_train_{classes_names[i]}.npy'))
    else:
        # Load the data from the train_val_numpy_arrays folder
        X_train = np.load(os.path.join(
            "train_val_numpy_arrays", f'X_train_biased_{classes_names[i]}.npy'))
        Y_train = np.load(os.path.join("train_val_numpy_arrays",
                                       f'y_train_biased_{classes_names[i]}.npy'))

    # add early stopping to the classifier to prevent overfitting
    clf = MLPClassifier(
        hidden_layer_sizes=best_hyperparams['hidden_layer_sizes'], max_iter=50, verbose=10, early_stopping=True)

    clf.fit(X_train, Y_train)

    joblib.dump(clf, os.path.join('models', classes_names[i] + '.joblib'))
