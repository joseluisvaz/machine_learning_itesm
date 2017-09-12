import numpy as np
import ML_Algorithms.LinearMethods as lm

class GradientClassifier(lm.LinearRegression):

    def fit(self,
            X,
            T,
            reg=0,
            step_size=0.001,
            max_iter=10000,
            tresh=0.0000001,
            step_type="fixed",
            coef_bias=0,
            print_val=False):

        """
        Implementation of Linear Regression using Gradient Descent
        """
        # Creating augmented Vector
        Xaug = self.augmentVector(X)

        def gradient_func(p): -2 * np.matmul(Xaug.T, T) + 2 * np.matmul(np.matmul(Xaug.T, Xaug), p) + reg * p

        def cost_function(p): np.dot((T - np.matmul(Xaug, p).reshape(T.shape)).T,
                                     (T - np.matmul(Xaug, p).reshape(T.shape))) + reg / 2.0 * np.dot(p, p)

        # Initializing the coefficients
        point = np.zeros(Xaug.shape[1]) + coef_bias

        self.coef = self.gradientDescent(cost_function, gradient_func, point, max_iter,
                                         tresh, step_size=step_size,
                                         step_type=step_type, print_val=print_val)
        return self

    def gradientDescent(self, cost_function, gradient_func, point, max_iter,
                        tresh, step_type="golden", step_size=0.0001, print_val=False):

        counter = 0
        self.list_coef = [point]

        while counter < max_iter:

            gradient = gradient_func(point)

            if step_type == "fixed":
                point = point - step_size * gradient

            elif step_type == "golden":
                point = point - self.goldenStep(cost_function, gradient_func, point) * gradient

            ###########################

            self.list_coef.append(point)

            if sum(abs(gradient)) < tresh:
                break

            counter = counter + 1

        if print_val:
            print("steps: " + str(counter))

        self.list_coef = np.array(self.list_coef)

        return point

    def goldenStep(self, function, gradient, point):
        def optimizer(s): function(point - s * gradient(point))
        return self.goldenSearch(optimizer)

    def goldenSearch(self, function, a=0, b=3, tresh=0.000001):

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
