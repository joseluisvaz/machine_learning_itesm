import numpy as np
import ML_Algorithms.Classifiers as lm
import ML_Algorithms.Optimization_methods as om


class GradientClassifier(lm.LinearRegression):

    def fit(self,
            X,
            T,
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
        Xaug = self.augmentVector(X)

        def gradient_func(p): return -2 * np.matmul(Xaug.T, T) + 2 * np.matmul(np.matmul(Xaug.T, Xaug), p) + reg * p

        def cost_function(p): return np.dot((T - np.matmul(Xaug, p).reshape(T.shape)).T,
                                            (T - np.matmul(Xaug, p).reshape(T.shape))) + reg / 2.0 * np.dot(p.T, p)

        # Initializing the coefficients
        point = np.random.randn(Xaug.shape[1], 1)

        self.coef, self.list_coef = om.gradientDescent(cost_function, gradient_func, point, max_iter, tresh,
                                                       step_size=step_size, step_type=step_type, print_val=print_val)
        return self


