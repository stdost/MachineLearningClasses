# Author: Stella Stoyanova
# Date:
# Project: CE3
# Acknowledgements: 
#

import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal
import numpy as np

def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    
    N, D = features.shape
    M = mu.shape[0]
    covariance_matrix = var * torch.eye(D)

    # Create a list to store the results
    phi = torch.zeros((N, M))

    # Loop over each mean vector 
    for i in range(M):
    # Create a multivariate normal distribution for each mean vector
        dist = multivariate_normal(mean=mu[i], cov=covariance_matrix)
        phi[:, i] = torch.from_numpy(dist.pdf(features))
    return phi

def _plot_mvn():
    fi = mvn_basis(X, mu, var)
    M = mu.shape[0]
    for i in range(M):
        plt.plot(fi[:,i])
    plt.xlabel("Data Points")
    plt.ylabel("Basis Function Value")
    #plt.savefig('2_1.png')
    plt.show()


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N, M = fi.shape
    
    # Compute (Phi^T Phi + lamda * I)
    regularization_term = lamda * torch.eye(M)
    FiT_Fi = torch.matmul(fi.T, fi)
    
    # Compute the MLE estimate for w
    wml = torch.linalg.solve(FiT_Fi + regularization_term, torch.matmul(fi.T, targets))
    
    return wml


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, var)
    
    # Step 2: Compute the predicted targets using the linear model
    predictions = torch.matmul(fi, w)
    
    return predictions


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """


