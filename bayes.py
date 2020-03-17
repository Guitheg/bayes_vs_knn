import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
        self.just_diago = False
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        # initalize the output vector
        y = np.empty(n_obs)
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        #constante a calculer une fois
        n_res = np.zeros((n_obs, n_classes))
        f_pi = n_features/2*np.log(2*np.pi)
        #print("priors:",self.priors)
        for i in range(n_classes):
            for k in range(n_obs):
                sigma_inv = np.linalg.inv(self.sigma[i])
                log_sigma_det = 1/2*np.log(abs(np.linalg.det(self.sigma[i])))

                n_res[k][i] = - 1/2 * np.dot((X[k] - self.mu[i]).T, np.dot(sigma_inv, (X[k]-self.mu[i])))- f_pi - log_sigma_det + np.log(self.priors[i])


        #print("predict")
        #print(n_res)
        y = np.argmax(n_res,axis=1)
        #print(y)
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        #print(X.shape)
        #print("test: \n",X[1])
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))
        if self.priors == None:
            self.priors = np.zeros((n_classes,1)) + 1/n_classes

        # calcul moyenne
        for i in range(n_classes):
            self.mu[i] = np.mean(X[y==i],axis = 0)
        #print(self.mu)
        
        # calcul covariance
        for i in range(n_classes):

            self.sigma[i] = np.cov(X[y == i].T)
            if self.just_diago:
                for j in range(n_features):
                    for k in range(n_features):
                        if(j!=k):
                            self.sigma[i][j][k] = 0
            
        #print(self.sigma)

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)





    def graph(self, X:np.ndarray, y:np.ndarray):

        R = X[:, 0]
        V = X[:, 1]

        n_obs = X.shape[0]
        vraisemblance = np.empty(n_obs)

        f_pi = 2/2*np.log(2*np.pi)
        print("priors:",self.priors)
        for k in range(n_obs):
            sigma_inv = np.linalg.inv(self.sigma[y[k]])
            log_sigma_det = 1/2*np.log(np.linalg.det(self.sigma[y[k]]))

            vraisemblance[k] = - 1/2 * np.dot((X[k] - self.mu[y[k]]).T, np.dot(sigma_inv, (X[k]-self.mu[y[k]])))- f_pi - log_sigma_det + np.log(self.priors[y[k]])

        couleur = ['red','blue','black']
        couleur_label = np.array([couleur[i] for i in y])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(R, V, vraisemblance, c=couleur_label)
        plt.show()

