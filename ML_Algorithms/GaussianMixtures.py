import numpy as np
import random


class UnivariateGaussian:
    """
    Univariate Gaussian
    """
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def pdf(self, datapoint):
        """
        Probability density function of univariate gaussian/normal distribution
        :param datapoint:
        :return:
        """
        z = (datapoint - self.mu) / abs(self.sigma)
        p_x = (1/(np.sqrt(2*self.sigma) * self.sigma))*np.exp(-z * z / 2)
        return p_x

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)


class MultivariateGaussian:
    """
    Implementation of Multivariate Gaussian Distribution
    """
    def __init__(self, mu, sigma):
        """
        Initialization of a Multivariate Gaussian distribution, we do precomputation to
        avoid extra computation in further methods.
        :param mu: Vertical vector of the mean
        :param sigma: Covariance Matrix
        """
        self.mu = mu
        self.sigma = sigma
        self.inv_sigma = np.linalg.pinv(sigma)
        self.det_sigma = np.linalg.det(sigma)

    def pdf(self, x):
        """
        Estimating probability at a single point
        :param x: datapoint must be the same size as mu
        :return: probability of this datapoint
        """
        if x.shape != (x.shape[0], 1):
            x = x.reshape((x.shape[0], 1))
        u = (x - self.mu)
        z = - 1/2.0 * np.dot(np.dot(u.T, self.inv_sigma), u)
        norm_factor = (1/((2*np.pi)**(self.mu.shape[0]/2.0)))*(1/np.sqrt(self.det_sigma))
        p_x = norm_factor * np.exp(z)

        return float(p_x)


class GaussianMixture:
    """
    Model Mixture of n multivariate Gaussian Distributions
    """

    def __init__(self,
                 data,
                 clusters=2,
                 ):
        """
        Random initialization of mean vector,
        Covariance Matrix initialized with Identity Matrix
        :param data: Data used by the parameters
        :param clusters: Number of clusters to fit
        """

        self.data = data
        self.clusters = clusters
        self.dists = {}
        self.mix = {}
        self.n = self.data.shape[0]
        self.d = self.data.shape[1]
        self.log_likelihood = 0
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        for i in range(clusters):
            self.dists["dist" + str(i)] = MultivariateGaussian(random.uniform(mins, maxs).reshape(mins.shape[0], 1),
                                                               np.identity(mins.shape[0]))
            self.mix["dist" + str(i)] = 1.0/self.clusters

    def expectation(self):
        """
        Expectation Step for Gaussian Mixtures
        :return: Parameters
        """
        # Initializing dictionary to store probabilities of n clusters
        prob_z = {}
        probs = {}

        self.log_likelihood = 0
        for i in range(self.clusters):
            probs["cluster" + str(i)] = np.zeros((self.data.shape[0], 1))

        for n in range(self.data.shape[0]):

            for i in range(self.clusters):
                prob_z["cluster" + str(i)] = self.dists["dist" + str(i)].pdf(self.data[n]) * self.mix["dist" + str(i)]

            # Normalization Factor
            den = sum(prob_z.values())
            # Normalize!
            sume = 0
            for i in range(self.clusters):
                prob_z["cluster" + str(i)] /= den
                sume += prob_z["cluster" + str(i)]

            self.log_likelihood += np.log(sume)

            for i in range(self.clusters):
                probs["cluster" + str(i)][n] = prob_z["cluster" + str(i)]

        return probs

    def maximization(self,  probs):

        # Updating the means
        accu_sigma = {}

        for i in range(self.clusters):

            den = np.sum(probs["cluster" + str(i)])
            self.dists["dist" + str(i)].mu = (np.sum(probs["cluster" + str(i)] * self.data, axis=0,
                                                     keepdims=True) / den).T

            accu_sigma["cluster" + str(i)] = np.zeros((self.d, self.d))

            for j in range(self.n):
                u = self.data[j].reshape((self.d, 1)) - self.dists["dist" + str(i)].mu
                accu_sigma["cluster" + str(i)] += probs["cluster" + str(i)][j] * np.dot(u, u.T)

            self.dists["dist" + str(i)].sigma = accu_sigma["cluster" + str(i)]/den
            self.mix["dist" + str(i)] = den / self.n

    def fit(self, iters=10, to_print=False):

        # List of Gaussians in each iteration

        if to_print:
            for i in range(self.clusters):
                mu = self.dists["dist" + str(i)].mu
                cov = self.dists["dist" + str(i)].sigma
                print("Iter 0 dist " + str(i + 1) + " mean: " + str(mu) + " covariance: " + str(cov))

        for j in range(1, iters + 1):
            self.maximization(self.expectation())
            if to_print:
                for i in range(self.clusters):
                    mu = self.dists["dist" + str(i)].mu
                    cov = self.dists["dist" + str(i)].sigma
                    print("Iter " + str(j) + " dist " + str(i + 1) + " mean: " + str(mu) + " covariance: " + str(cov))
            else:
                if j == iters:
                    for i in range(self.clusters):
                        mu = self.dists["dist" + str(i)].mu
                        cov = self.dists["dist" + str(i)].sigma
                        print(" dist " + str(i + 1))
                        print(" mean: ")
                        print(mu)
                        print(" covariance: ")
                        print(cov)