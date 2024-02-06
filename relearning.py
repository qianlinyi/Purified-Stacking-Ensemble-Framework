import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns


def draw(y_true, y_pred, classes: int):
    """
    draw confusion matrix
    :param y_true: true labels
    :param y_pred: predicted labels
    :param classes: number of classes
    :return:
    """
    m = confusion_matrix(y_true, y_pred)
    xtick = [_ for _ in range(classes)]
    ytick = [_ for _ in range(classes)]
    sns.heatmap(m, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=xtick, yticklabels=ytick, square=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.savefig('cf.png')
    plt.show()


def relearning(x_train, y_train, x_test, y_test, correct_preds, correct_labels, classes):
    meta_learner = LogisticRegression()
    meta_learner.fit(
        X=x_train,
        y=y_train
    )
    if len(x_test):
        preds = meta_learner.predict(x_test)
        for i in range(len(x_test)):
            if preds[i] != y_test[i]:
                print(f'wrong sample:\n{x_test[i]}, {y_test[i]}')
        correct_preds += list(preds)
        correct_labels += list(y_test)
        print(f'Accuracy: {accuracy_score(correct_labels, correct_preds)}')
        print(f'Recall: {np.average(recall_score(correct_labels, correct_preds, average=None))}')
        print(f'Precision: {np.average(precision_score(correct_labels, correct_preds, average=None))}')
        print(f'F1: {np.average(f1_score(correct_labels, correct_preds, average=None))}')
        print(f'Confusion Matrix:\n{confusion_matrix(correct_labels, correct_preds)}')

        draw(correct_labels, correct_preds, classes)
