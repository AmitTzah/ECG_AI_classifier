from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import joblib

# Load the data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')


if 'best_hyperparams_mlp.txt' not in os.listdir():
    best_hyperparams = {'hidden_layer_sizes': (100, 100)}

else:

    with open('best_hyperparams_mlp.txt', 'r') as f:
        best_hyperparams = f.read()

    # convert the string to a dictionary

    best_hyperparams = eval(best_hyperparams)


# Partition the problem into 7 different single label classes
classifiers = []
for i in range(7):
    y = Y_train[:, i]

    X_train_biased = X_train
    # augment the data so it is biased towards positive examples, augment by 5 times
    for j in range(len(y)):
        if y[j] == 1:
            for k in range(5):
                X_train_biased = np.append(
                    X_train_biased, [X_train[j]], axis=0)
                y = np.append(y, [1], axis=0)

    # define the classifier us
    clf = MLPClassifier(
        hidden_layer_sizes=best_hyperparams['hidden_layer_sizes'], max_iter=50, verbose=10)
    clf.fit(X_train_biased, y)
    classifiers.append(clf)


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

# save the models in the models folder

for i, clf in enumerate(classifiers):

    # save the model using joblib

    joblib.dump(clf, os.path.join('models', classes_names[i] + '.joblib'))
