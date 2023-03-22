from joblib import dump, load
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

classes_names = ["sinus rhythm", "myocardial infarction", "left axis deviation",
                 "abnormal QRS", "left ventricular hypertrophy", "t wave abnormal", "myocardial ischemia", "other"]


# load all the mlp models from the models folder
classifiers = []

for file in os.listdir('models'):
    clf = load(os.path.join('models', file))

    # append the model to the classifiers list in the same order as the classes_names list
    if file == 'sinus rhythm.joblib':
        classifiers.append((clf, '0'))
    elif file == 'myocardial infarction.joblib':
        classifiers.append((clf, '1'))
    elif file == 'left axis deviation.joblib':
        classifiers.append((clf, '2'))
    elif file == 'abnormal QRS.joblib':
        classifiers.append((clf, '3'))
    elif file == 'left ventricular hypertrophy.joblib':
        classifiers.append((clf, '4'))
    elif file == 't wave abnormal.joblib':
        classifiers.append((clf, '5'))
    elif file == 'myocardial ischemia.joblib':
        classifiers.append((clf, '6'))

# sort the classifiers list by the second element in the tuple
classifiers.sort(key=lambda x: x[1])


X_test = np.load('train_val_numpy_arrays/X_val.npy')
Y_test = np.load('train_val_numpy_arrays/Y_val.npy')


# remove the other class from Y_test (the last column)
Y_test = np.delete(Y_test, 7, axis=1)

# make a classification_reports folder if it doesn't exist
if 'classification_reports' not in os.listdir():
    os.mkdir('classification_reports')

# define y_pred list with the correct shape
Y_pred = np.zeros((len(Y_test), 7))

# predict the labels of the test data using the mlp models
for i in range(7):

    # define the coorect labels for the current class
    Y_test_class = Y_test[:, i]
    Y_pred_class = classifiers[i][0].predict(X_test)

    # add it to the y_pred list
    Y_pred[:, i] = Y_pred_class

    # prrint the accuracy score, f1 score, precision score and recall score
    print('Accuracy score for ' +
          classes_names[i] + ' is: ' + str(accuracy_score(Y_test_class, Y_pred_class)))
    print('F1 score for ' +
          classes_names[i] + ' is: ' + str(f1_score(Y_test_class, Y_pred_class)))
    print('Precision score for ' +
          classes_names[i] + ' is: ' + str(precision_score(Y_test_class, Y_pred_class)))
    print('Recall score for ' +
          classes_names[i] + ' is: ' + str(recall_score(Y_test_class, Y_pred_class)))

    # write these scores to a text file in the classification_reports folder for the current class
    with open('classification_reports\classification_report_' + classes_names[i] + '.txt', 'w') as f:
        f.write('Accuracy score:' + str(
            accuracy_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('F1 score: ' + str(
            f1_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('Precision score: ' +
                str(precision_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('Recall score: ' +
                str(recall_score(Y_test_class, Y_pred_class)) + ' \n')


confusion_matrix = multilabel_confusion_matrix(Y_test, Y_pred)


fig, axes = plt.subplots(4, 2, figsize=(10, 10))
axes = axes.ravel()
for i, ax in enumerate(axes):

    if i == 7:
        # remove the last subplot
        fig.delaxes(ax)
        break

    cm = confusion_matrix[i]
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(classes_names[i])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.xaxis.set_ticklabels(['', '0', '1'])
    ax.yaxis.set_ticklabels(['', '0', '', '1'])

    # set the color of the text in the confusion matrix
    cthresh = cm.max() / 1.1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > cthresh else "black")

plt.tight_layout()
plt.show()
