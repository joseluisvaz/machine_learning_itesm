import numpy as np


def true_pos(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1.0 and Ypred[i] == 1.0:
            count += 1
    return count


def true_neg(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 0.0 and Ypred[i] == 0.0:
            count += 1
    return count


def false_pos(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 0.0 and Ypred[i] == 1.0:
            count += 1
    return count


def false_neg(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1.0 and Ypred[i] == 0.0:
            count += 1
    return count


def confusion_matrix(Y, Ypred):
    conf = np.zeros((2, 2))
    conf[0, 0] = true_pos(Y, Ypred)
    conf[0, 1] = false_pos(Y, Ypred)
    conf[1, 0] = false_neg(Y, Ypred)
    conf[1, 1] = true_neg(Y, Ypred)
    return conf
