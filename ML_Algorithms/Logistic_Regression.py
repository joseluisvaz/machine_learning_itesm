import numpy as np
from ML_Algorithms import Classifiers
from ML_Algorithms import utils
from ML_Algorithms import Optimization_methods as om
from ML_Algorithms import metrics as mtr


class Logistic_Regression(Classifiers.LinearRegression):


    def fit(self,
            X,
            T,
            solver="gradient",
            reg=0,
            step_size=0.001,
            max_iter=100,
            tresh=0.00001,
            step_type="fixed",
            print_val=False):

        """
        Implementation of Linear Regression using Gradient Descent
        """

        # Creating augmented Vector
        Xaug = utils.augment_vector(X)

        m = Xaug.shape[0]
        d = Xaug.shape[1]

        if solver == "gradient":
            step_type = "fixed"
        elif solver == "gradient-gold":
            step_type = "golden"

        def cost_function(p):
            A = utils.sigmoid(np.dot(Xaug, p))
            inner = T * np.log(A) + (1 - T) * np.log(1 - A)
            print(inner)
            cost = -1 / m * np.sum(T * np.log(A) + (1 - T) * np.log(1 - A)) + reg / 2.0 * np.dot(p.T, p)
            return cost

        def gradient_func(p):
            grad = np.zeros(p.shape)
            u = utils.sigmoid(np.dot(Xaug, p)) - T
            for i in range(d):
                z = np.multiply(u, Xaug[:, i].reshape((Xaug.shape[0], 1)))
                grad[i] = np.sum(z) + reg * p[i]
            return grad

        # Initializing the coefficients
        point = np.random.randn(Xaug.shape[1], 1)

        self.coef, self.list_coef = om.gradientDescent(cost_function, gradient_func, point, max_iter, tresh,
                                                       step_size=step_size, step_type=step_type, print_val=print_val)
        return self

    def predict(self, X):
        X = utils.augment_vector(X)
        u = np.dot(X, self.coef)
        Y = utils.sigmoid(u)
        for i in range(Y.shape[0]):
            if Y[i] >= 0.5:
                Y[i] = 1
            elif Y[i] < 0.5:
                Y[i] = 0
        return Y

    def roc(self, X, Y):

        # Creating augmented  Vector
        Xaug = utils.augment_vector(X)

        # Linear Regression
        Ypred = utils.sigmoid(np.dot(Xaug, self.coef))
        x = np.linspace(0.01, 0.99, 99)
        pred_label = np.zeros(Y.shape)
        ROC = []
        for j in x:
            for i in range(Y.shape[0]):
                if Ypred[i] >= j:
                    pred_label[i] = 1
                elif Ypred[i] < j:
                    pred_label[i] = 0
            ROC.append([1 - mtr.specificity(Y, pred_label),
                        mtr.sensitivity(Y, pred_label)])
        return np.array(ROC)