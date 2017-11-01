import numpy as np


def true_pos(Y, Ypred):
    if -1.0 in Y:
        val = -1.0
    else:
        val = 0

    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1.0 and Ypred[i] == 1.0:
            count += 1
    return float(count)


def true_neg(Y, Ypred):
    if -1.0 in Y:
        val = -1.0
    else:
        val = 0

    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == val and Ypred[i] == val:
            count += 1
    return float(count)


def false_pos(Y, Ypred):
    if -1.0 in Y:
        val = -1.0
    else:
        val = 0

    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == val and Ypred[i] == 1.0:
            count += 1
    return float(count)


def false_neg(Y, Ypred):
    if -1.0 in Y:
        val = -1.0
    else:
        val = 0

    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1.0 and Ypred[i] == val:
            count += 1
    return float(count)


def accuracy(Y, Ypred):
    return (true_pos(Y, Ypred) + true_neg(Y, Ypred))/Y.shape[0]


def sensitivity(Y, Ypred):
    return true_pos(Y, Ypred)/(true_pos(Y, Ypred) + false_neg(Y, Ypred))


def specificity(Y, Ypred):
    return true_neg(Y, Ypred)/(true_neg(Y, Ypred) + false_pos(Y, Ypred))


def precision(Y, Ypred):
    return true_pos(Y, Ypred)/(true_pos(Y, Ypred) + false_pos(Y, Ypred))


def confusion_matrix(Y, Ypred):
    conf = np.zeros((2, 2))
    conf[0, 0] = int(true_pos(Y, Ypred))
    conf[0, 1] = int(false_pos(Y, Ypred))
    conf[1, 0] = int(false_neg(Y, Ypred))
    conf[1, 1] = int(true_neg(Y, Ypred))
    return conf


def f1_score(Y, Ypred):
    nom = precision(Y, Ypred) * sensitivity(Y, Ypred)
    den = precision(Y, Ypred) + sensitivity(Y, Ypred)
    return 2 * (nom / den)


def print_metrics(Y, Ypred, name):
    print(name)
    print("Conf_matrix:")
    print(confusion_matrix(Y, Ypred))
    print("Accuracy:")
    print(accuracy(Y, Ypred))
    print("F1_score:")
    print(f1_score(Y, Ypred))