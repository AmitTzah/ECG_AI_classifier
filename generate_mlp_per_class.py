from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import joblib

# Load the data
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

# Partition the problem into 7 different single label classes
classifiers = []
for i in range(7):
    y = Y_train[:, i]
    clf = MLPClassifier(hidden_layer_sizes=(
        100, 50), max_iter=75, verbose=10)
    clf.fit(X_train, y)
    classifiers.append(clf)


classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]

# Check if a "models" folder exists, if not create one
# if it does, remove all the files in it and then create it

if 'models' in os.listdir():

    for file in os.listdir('models'):

        # remove all the files in the models folder

        os.remove(os.path.join('models', file))

    # create the models folder
    os.mkdir('models')

else:
    os.mkdir('models')

# save the models in the models folder

for i, clf in enumerate(classifiers):

    # save the model using joblib

    joblib.dump(clf, os.path.join('models', classes_names[i] + '.joblib'))
