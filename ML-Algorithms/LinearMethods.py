"""
This is a class for Linear Classification and Regression

The canonical solution of linear regression with regularization is implemented

This method is also wrapped with Linear Classiier and Linear Machine with their respective discriminant function

"""

import numpy as np

class LinearRegression(object):


    def __init__(self):
        self.coef = None
        self.dim = None

    def augmentVector(self, X):
        """
        Function for augmenting Vector [numpy.array]

        :param X: numpy array
        :return: augmented numpy array
        """

        Xdummy = np.ones((X.shape[0], 1))
        X = np.concatenate([Xdummy, X], axis=1)

        return X

    def fit(self, X, T, reg=0):
        """
        This method uses the canonical solution of linear regression to solve for Xw = Y

        :param X: Input Data set as a numpy array
        :param T: Labels Vector as a numpy array
        :param reg:  Regularization constant
        :return: self
        """

        # Creating augmented Vector
        Xaug = self.augmentVector(X)

        # Solution with Regularization
        Xinner = np.matmul(np.linalg.inv(reg * np.identity(Xaug.shape[1]) + np.matmul(Xaug.T, Xaug)), Xaug.T)
        self.coef = np.matmul(Xinner, T)
        self.dim = self.coef.shape

        return self

    def predict(self, X):
        """
        Discrimination function for classifying samples:
        :param Y:
        :return:
        """

        # Creating augmented  Vector
        Xaug = self.augmentVector(X)

        Y = np.matmul(Xaug, self.coef)

        return Y


class LinearClassifier(LinearRegression):

    def predict(self, X):
        """
        Discrimination function for classifying samples

        :param X: numpy.array with samples
        :return: Y prediction
        """
        # Creating augmented  Vector
        Xaug = self.augmentVector(X)

        # Linear Regression
        Y = np.matmul(Xaug, self.coef)

        # Discriminant Function
        for i in range(Y.shape[0]):
            if Y[i] >= 0:
                Y[i] = 1
            elif Y[i] < 0:
                Y[i] = -1

        return Y

class LinearMachine(LinearRegression):

    def predict(self, X):
        """
        Discrimination function for classifying samples

        :param X: numpy.array with samples
        :return: T prediction
        """

        # Creating augmented  Vector
        Xaug = self.augmentVector(X)

        # Linear Regression
        T = np.matmul(Xaug, self.coef)

        # Discrimination Function
        prediction = np.zeros(T.shape)
        for i in range(T.shape[0]):
            prediction[i][T[i].argmax()] = 1

        return prediction
