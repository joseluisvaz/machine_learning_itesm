import numpy as np
import ML_Algorithms.Classifiers as lm
from math import sqrt


class FisherDiscriminant(lm.LinearRegression):

    def fit(self, X, T):

        """
        Implementation of Fisher Linear Discriminant
        """
        # Creating augmented Vector
        Xaug = self.augmentVector(X)



        self.coef = coef
        return self
