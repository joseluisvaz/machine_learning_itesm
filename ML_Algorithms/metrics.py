import numpy as np

def true_pos(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1 and Ypred[i] == 1:
            count += 1
    return count

def true_neg(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1 and Ypred[i] == 1:
            count += 1
    return count

def false_pos(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1 and Ypred[i] == 1:
            count += 1
    return count

def false_neg(Y, Ypred):
    count = 0
    for i in range(Y.shape[0]):
        if Y[i] == 1 and Ypred[i] == 1:
            count += 1
    return count

def confussion_matrix(Y,Ypred): return 0
