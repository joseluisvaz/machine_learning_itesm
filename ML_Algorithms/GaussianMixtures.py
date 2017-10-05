import numpy as np
import math
import random


class UnivariateGaussian:
    "Univariate Gaussian"
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def pdf(self, datapoint):
        "Probability dennsity function of univariate gaussian/normal distribution"
        z = (datapoint - self.mu)/ abs(self.sigma)
        p_x = (1/(np.sqrt(2*self.sigma) * self.sigma))*np.exp(-z * z /2)
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
        u = (x - self.mu)
        z = - 1/2.0 * np.dot(np.dot(u.T, self.inv_sigma), u)
        norm_factor = (1/((2*np.pi)**(self.mu.shape[0]/2.0)))*(1/np.sqrt(self.det_sigma))
        p_x = norm_factor * np.exp(z)

        return p_x

class GaussianMixture:
    """
    Model Mixture of n multivariate Gaussian Distributions
    """

    def __init__(self,
                 data,
                 clusters = 2,
                 sigma_range = (0.1, 1),
                 mix = 0.5
                 ):
        """
        Random initialization of parameters within the give ranges

        :param data: Data used by the parameters
        :param clusters: Number of clusters to fit the data to
        :param sigma_range: Range of sigma values.
        :param mix: Parameter to mix the data.
        """

        self.data = data
        self.dists = {}
        self.mix = mix
        mu_range = (min(data), max(data))
        for i in range(clusters):
            self.dists["dist" + str(i)] = UnivariateGaussian(random.uniform(mu_range[0], mu_range[1]),
                                                             random.uniform(sigma_range[0], sigma_range[1]))






