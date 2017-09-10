import numpy as np
import ML_Algorithms.LinearMethods as lm
from math import sqrt

class GradientClassifier(lm.LinearRegression):

    def goldenSearch(self,f , a=0, b=1, tresh=0.001):

        golden_ratio = 0.618034

        #Define initial length for search
        length = b - a

        lambda_1 = a + golden_ratio**2 * length
        lambda_2 = a + golden_ratio * length

        while length > tresh:

            if f(lambda_1) > f(lambda_2):
                a = lambda_1
                lambda_1 = lambda_2
                length = b - a
                lambda_2 = a + golden_ratio * length
            else:
                b = lambda_2
                lambda_2 = lambda_1
                length = b - a
                lambda_1 = a + golden_ratio**2 * length

        return (b+a)/2.0

    def fit(self,
            X,
            T,
            reg=0,
            step_size=0.001,
            max_iter=10000,
            tresh=0.0000001,
            step="fixed",
            print_iter=False):

        """
        Implementation of Gradient Descent
        """
        #Creating augmented Vector
        Xaug = self.augmentVector(X)

        #Initializing the coefficients
        coef = np.zeros(Xaug.shape[1])
        grad = np.zeros(Xaug.shape[1])

        counter = 0
        while(counter < max_iter):

            #iterating through columns
            loss = T - np.matmul(Xaug, coef).reshape(T.shape)
            for j in range(Xaug.shape[1]):
                grad[j] = - np.matmul(loss.T, Xaug[:, j]) + reg*coef[j]

            #updating after the gradient is calculate
            #grad = grad/Xaug.shape[0]
            grad = grad/np.linalg.norm(grad)

            if step == "fixed":
                coef = coef - step_size*grad
            elif step == "golden":
                f = lambda x: np.dot((T - np.matmul(Xaug, coef + x * grad).reshape(T.shape)).T,
                                (T - np.matmul(Xaug, coef + x * grad).reshape(T.shape)))
                stepie = self.goldenSearch(f)
                coef = coef - stepie*grad

            if print_iter and counter%100 == 0:
                print('coef:' + str(coef))

            if sum(abs(grad)) < tresh:
                print("steps: " + str(counter))
                break

            counter = counter + 1

        self.coef = coef
        return self



