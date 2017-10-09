import numpy as np


def augment_vector(X):
    """
    Function for augmenting Vector [numpy.array]

    :param X: numpy array
    :return: augmented numpy array
    """

    Xdummy = np.ones((X.shape[0], 1))
    X = np.concatenate([Xdummy, X], axis=1)

    return X

