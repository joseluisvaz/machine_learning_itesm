"""
This is a class for Linear Classification and Regression

The canonical solution of linear regression with regularization is implemented

This method is also wrapped with Linear Classifier and Linear Machine with their respective discriminant function

"""

import numpy as np
import ML_Algorithms.utils as utils
import ML_Algorithms.Optimization_methods as om


class LinearRegression(object):

    def __init__(self):
        self.coef = None
        self.dim = None
        self.list_coef = []

    def fit(self,
            X,
            T,
            solver="gradient",
            reg=0,
            step_size=0.001,
            max_iter=10000,
            tresh=0.00001,
            step_type="fixed",
            print_val=False):

        """
        Implementation of Linear Regression using Gradient Descent
        """

        # Creating augmented Vector
        Xaug = utils.augment_vector(X)

        if solver == "canonical":
            Xinner = np.matmul(np.linalg.inv(reg * np.identity(Xaug.shape[1]) + np.matmul(Xaug.T, Xaug)), Xaug.T)
            self.coef = np.matmul(Xinner, T)
            self.dim = self.coef.shape

        else:
            if solver == "gradient":
                step_type = "fixed"
            elif solver == "gradient-gold":
                step_type = "golden"

            def gradient_func(p): return -2 * np.matmul(Xaug.T, T) + 2 * np.matmul(np.matmul(Xaug.T, Xaug), p) + reg * p

            def cost_function(p): return np.dot((T - np.matmul(Xaug, p).reshape(T.shape)).T,
                                                (T - np.matmul(Xaug, p).reshape(T.shape))) + reg / 2.0 * np.dot(p.T, p)

            # Initializing the coefficients
            point = np.random.randn(Xaug.shape[1], 1)

            self.coef, self.list_coef = om.gradientDescent(cost_function, gradient_func, point, max_iter, tresh,
                                                           step_size=step_size, step_type=step_type, print_val=print_val)

        return self

    def predict(self, X):
        """
        Discrimination function for classifying samples:
        :param Y:
        :return:
        """

        # Creating augmented  Vector
        Xaug = utils.augment_vector(X)

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
        Xaug = utils.augment_vector(X)

        # Linear Regression
        Y = np.matmul(Xaug, self.coef)

        # Discriminant Function
        for i in range(Y.shape[0]):
            if Y[i] >= 0.5:
                Y[i] = 1
            elif Y[i] < 0.5:
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
        Xaug = utils.augment_vector(X)

        # Linear Regression
        T = np.matmul(Xaug, self.coef)

        # Discrimination Function
        prediction = np.zeros(T.shape)
        for i in range(T.shape[0]):
            prediction[i][T[i].argmax()] = 1

        return prediction
