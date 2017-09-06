import numpy as np
import ML_Algorithms.LinearMethods as lm

class GradientClassifier(lm.LinearRegression):

    def fit(self, X, T, reg=0, step_size = 0.01, max_iter = 10000, tresh = 0.0000001):
        """
        This method fits a classifier using the method of gradient descent
        :param X: Input Data set as a numpy array
        :param T: Labels Vector as a numpy  array
        :param reg: Parameter for regularization
        :return: self
        """

        #Creating augmented Vector
        Xaug = self.augmentVector(X)

        #Initializing the coefficients
        coef = np.zeros(Xaug.shape[1]) + 0.01
        grad = np.zeros(Xaug.shape[1])

        while(max_iter > 0):
            #iterating through columns
            loss = T - np.matmul(Xaug, coef).reshape(T.shape)
            for j in range(Xaug.shape[1]):
                grad[j] = - np.matmul(loss.T, Xaug[:, j]) + reg*coef[j]

            #updating after the gradient is calculated
            grad = grad/Xaug.shape[0]
            factor = step_size*grad
            coef = coef - factor

            if abs(sum(factor)) < tresh:
                break

            max_iter = max_iter - 1

        self.coef = coef
        return self






