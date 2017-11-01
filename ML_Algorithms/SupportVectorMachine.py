import numpy as np
import pandas as pd
import cvxopt
import metrics as mtr


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMTrainer(object):
    def __init__(self, kernel, c):
        """
        :param kernel: kernel function from Kernel class
        :param c: Hyperparameter for the soft margin
        """
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        """
        Returns  a predictor with the support vectorsd
        :param X: Training data matrix
        :param y: Training data matrix
        :return:  SVM predictor
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        """
        Construct the predictor
        :param X: Data matrix
        :param y: Labels vector
        :param lagrange_multipliers: List of the lagrange multipliers
        :return: The support vector machine predictor
        """

        # Filter the multipliers, the ones close to 0 do not belong to the suppport vectoors
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # Get the bias term of the predictor
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, X, y):
        """
        This method maximizes the quadratic programming problem of
        L = 1/2 * x^T P x + q^T x subject to Gx <= h and Ax = b
        Which for the support vector machine problem is maximizing the negative hinge loss function
        L = 1/2 * a^T Q a - np.sum(a) subject to 0 <= a <= C and y^T a = 0
        where Qij = yi yj K(xi, xj) => Q = outer(y,y) * gram_matrix

        :param X: Data Matrix
        :param y: Labels [1,-1]
        :return: lagrange multipliers
        """
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i < = 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i < = c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        # Stack contraints vertically to satisfy 0 <= a <= C
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # return a list of lagrange multipliers
        return np.ravel(solution['x'])

    def _compute_multipliers_nu(self, X, y):
        """
        This method maximizes the quadratic programming problem of
        L = 1/2 * x^T P x + q^T x subject to Gx <= h and Ax = b
        Which for the nu - support vector machine (Schoelkopf et al. 2000)  is maximizing the negative hinge loss function
        L = 1/2 * a^T Q a  subject to 0 <= a <= 1/N , y^T a = 0 and np.sum(a) >= nu
        where Qij = yi yj K(xi, xj) => Q = outer(y,y) * gram_matrix

        :param X: Data Matrix
        :param y: Labels [1,-1]
        :return: lagrange multipliers
        """
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.zeros(n_samples))

        # -a_i < = 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i < = 1/n
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * 1/n_samples)

        # np.sum(a) >= 1
        G_nu = cvxopt.matrix(np.ones(1, n_samples))
        h_nu = cvxopt.matrix(self._c)

        # Stack contraints vertically to satisfy 0 <= a <= C
        G = cvxopt.matrix(np.vstack((G_std, G_slack, G_nu)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack, h_nu)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # return a list of lagrange multipliers
        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

    def predict_all(self, X):
        """
        Computes the SVM prediction for the data X.
        """
        prediction = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            prediction[i] = self.predict(x)

        return prediction


    def roc(self, X, Y):

        biases = np.linspace(self._bias - 10, self._bias + 10, 50)
        ROC = []
        for i, bias in enumerate(biases):
            predictor = SVMPredictor(
                kernel=self._kernel,
                bias=bias,
                weights=self._weights,
                support_vectors=self._support_vectors,
                support_vector_labels=self._support_vector_labels)

            Ypred = np.zeros(X.shape[0])
            for j, x in enumerate(X):
                Ypred[j] = predictor.predict(x)

            ROC.append([1 - mtr.specificity(Y, Ypred),
                        mtr.sensitivity(Y, Ypred)])

        return np.array(ROC)


