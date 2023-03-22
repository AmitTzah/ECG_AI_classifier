from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import joblib

if 'best_hyperparams_mlp.txt' not in os.listdir():
    best_hyperparams = {'hidden_layer_sizes': (100, 50, 50),
                        'activation': 'logistic',
                        'learning_rate_init': 0.01,
                        'early_stopping': True,
                        'tol': 0.00001,

                        }


else:

    with open('best_hyperparams_mlp.txt', 'r') as f:
        best_hyperparams = f.read()

    # convert the string to a dictionary

    best_hyperparams = eval(best_hyperparams)

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


if 'models' in os.listdir():

    for file in os.listdir('models'):

        # remove all the files in the models folder

        os.remove(os.path.join('models', file))


else:
    os.mkdir('models')


# Partition the problem into 7 different single label classes
for i in range(7):

    # Load the data from the train_val_numpy_arrays folder
    X_train = np.load(os.path.join(
        "train_val_numpy_arrays", f'X_train_balanced_{classes_names[i]}.npy'))
    Y_train = np.load(os.path.join("train_val_numpy_arrays",
                                   f'y_train_balanced_{classes_names[i]}.npy'))

    # add early stopping to the classifier to prevent overfitting
    clf = MLPClassifier(
        hidden_layer_sizes=best_hyperparams['hidden_layer_sizes'],
        activation=best_hyperparams['activation'],
        learning_rate_init=best_hyperparams['learning_rate_init'],
        tol=best_hyperparams['tol'],

        max_iter=75, verbose=10, early_stopping=True)

    clf.fit(X_train, Y_train)

    joblib.dump(clf, os.path.join('models', classes_names[i] + '.joblib'))
