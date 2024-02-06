import numpy as np


def purify(x, y, classes, classifier_num):
    new_x, new_y, correct_preds, correct_labels = [], [], [], []
    for i, sample in enumerate(x):
        preds = np.split(sample, classifier_num)  # obtain output of each base learner
        one_hot_label = np.zeros(shape=(classes, ))
        one_hot_label[int(y[i])] = 1
        distances = np.array([np.linalg.norm(pre-one_hot_label) for pre in preds])
        weights = np.array([np.divide(1-dis/np.sum(distances), classifier_num-1) for dis in distances])
        ensemble = np.sum(np.array([np.dot(weights[_], preds[_]) for _ in range(classifier_num)]), axis=0)
        label = np.argmax(ensemble)
        if label != y[i]:  # conflict
            new_x.append(sample)
            new_y.append(y[i])
        else:
            correct_preds.append(label)
            correct_labels.append(y[i])
    return correct_preds, correct_labels, np.array(new_x), np.array(new_y)


def purification(x_train, y_train, x_test, y_test, classes, classifier_num):
    _, _, new_x_train, new_y_train = purify(x_train, y_train, classes, classifier_num)
    print(f'size of purified training set for meta-learner: {new_x_train.shape}, {new_y_train.shape}')

    correct_preds, correct_labels, new_x_test, new_y_test = purify(x_test, y_test, classes, classifier_num)
    print(f'size of purified testing set for meta-learner: {new_x_test.shape}, {new_y_test.shape}')

    return new_x_train, new_y_train, new_x_test, new_y_test, correct_preds, correct_labels
