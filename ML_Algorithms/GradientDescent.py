import numpy as np
import ML_Algorithms.LinearMethods as lm
from math import sqrt


class GradientClassifier(lm.LinearRegression):

    def fit(self,
            X,
            T,
            reg=0,
            step_size=0.001,
            max_iter=10000,
            tresh=0.0000001,
            step="fixed",
            coef_bias = 0,
            print_iter=False):

        """
        Implementation of Gradient Descent
        """
        # Creating augmented Vector
        Xaug = self.augmentVector(X)

        # Initializing the coefficients
        coef = np.zeros(Xaug.shape[1]) + coef_bias
        grad = np.zeros(Xaug.shape[1])

        counter = 0
        self.list_coef = [coef]
        while counter < max_iter:

            # iterating through columns
            loss = T - np.matmul(Xaug, coef).reshape(T.shape)

            grad[0] = - np.matmul(loss.T, Xaug[:, 0])
            for j in range(1, Xaug.shape[1]):
                grad[j] = - np.matmul(loss.T, Xaug[:, j]) + reg * coef[j]


            if step == "fixed":
                coef = coef - step_size * grad
            elif step == "golden":
                grad = grad / np.linalg.norm(grad)
                function = lambda x: np.dot((T - np.matmul(Xaug, coef - x * grad).reshape(T.shape)).T,
                                     (T - np.matmul(Xaug, coef - x * grad).reshape(T.shape)))
                stepie = self.goldenSearch(function, a=0, b=3)
                coef = coef - stepie * grad

            self.list_coef.append(coef)

            if print_iter and counter % 100 == 0:
                print('coef:' + str(coef))

            if sum(abs(grad)) < tresh and print_iter:
                print("steps: " + str(counter))
                break

            counter = counter + 1

        self.list_coef = np.array(self.list_coef)

        self.coef = coef
        return self

    def goldenSearch(self, function, a=0, b=1, tresh=0.000001):

        golden_ratio = 0.618034

        # Define initial length for search
        length = b - a

        lambda_1 = a + golden_ratio ** 2 * length
        lambda_2 = a + golden_ratio * length

        while length > tresh:

            if function(lambda_1) > function(lambda_2):
                a = lambda_1
                lambda_1 = lambda_2
                length = b - a
                lambda_2 = a + golden_ratio * length
            else:
                b = lambda_2
                lambda_2 = lambda_1
                length = b - a
                lambda_1 = a + golden_ratio ** 2 * length

        return (b + a) / 2.0
