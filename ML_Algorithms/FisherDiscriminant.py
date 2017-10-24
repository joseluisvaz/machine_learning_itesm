import numpy as np
import ML_Algorithms.Classifiers as lm
import ML_Algorithms.utils as utils
from math import sqrt


class FisherDiscriminant(lm.LinearRegression):


    # TODO: Fisher linear discriminant implementation
    def fit(self, X, T):

        """
        Implementation of Fisher Linear Discriminant
        """
        # Creating augmented Vector
        Xaug = utils.augment_vector(X)

        self.coef = coef
        return self
