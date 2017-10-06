import numpy as np
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
                 mix=(0.3333, 0.3333, 0.3333)
                 ):
        """
        Random initialization of mean vector,
        Covariance Matrix initialized with Identity Matrix
        :param data: Data used by the parameters
        :param clusters: Number of clusters to fit the data to
        :param mix: Parameter to mix the data.
        """

        self.data = data
        self.clusters = clusters
        self.dists = {}
        self.mix = mix
        self.log_likelihood = 0
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        for i in range(clusters):
            self.dists["dist" + str(i)] = MultivariateGaussian(random.uniform(mins, maxs).reshape(mins.shape[0], 1),
                                                               np.identity(mins.shape[0]))

    def expectation(self):
        """
        Expectation Step for Gaussian Mixtures
        :return: Parameters
        """
        # Initializing dictionary to store probabilities of n clusters
        prob_z = {}
        probs = np.zeros(self.data[0], 1)

        for i in range(self.clusters):
            probs["cluster" + str(i)] = np.zeros((self.data.shape[0], 1))

        for n in range(self.data.shape[0]):

            for i in range(self.clusters):
                prob_z["cluster" + str(i)] = self.dists["dist" + str(i)].pdf(self.data[n]) * self.mix[0]

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







