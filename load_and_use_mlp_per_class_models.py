from joblib import dump, load
from sklearn.metrics import confusion_matrix

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

# if X_test.npy and Y_test.npy exist, load them
if 'X_test.npy' in os.listdir() and 'Y_test.npy' in os.listdir():
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')


# make a classification_reports folder if it doesn't exist
if 'classification_reports' not in os.listdir():
    os.mkdir('classification_reports')


# define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# predict the labels of the test data using the mlp models
for i in range(7):
    # define the coorect labels for the current class
    Y_test_class = Y_test[:, i]
    Y_pred_class = classifiers[i][0].predict(X_test)
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
        f.write('Accuracy score for ' + classes_names[i] + ' is: ' + str(
            accuracy_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('F1 score for ' + classes_names[i] + ' is: ' +
                str(f1_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('Precision score for ' + classes_names[i] + ' is: ' + str(
            precision_score(Y_test_class, Y_pred_class)) + ' \n')
        f.write('Recall score for ' + classes_names[i] + ' is: ' + str(
            recall_score(Y_test_class, Y_pred_class)) + ' \n')

    # save the confusion matrix for the current class
    # no need to use multilabel_confusion_matrix because the current class is binary
    cm = confusion_matrix(Y_test_class, Y_pred_class)
    # save the confusion matrix as a png file
    plt.figure()
    plot_confusion_matrix(cm, classes=[classes_names[i]],
                          title='Confusion matrix, without normalization')
    plt.savefig('classification_reports\confusion_matrix_' +
                classes_names[i] + '.png')
    plt.close()
