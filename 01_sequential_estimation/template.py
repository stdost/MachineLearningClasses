# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov = (var**2)*np.identity(k)  # Diagonal covariance matrix
    samples = np.random.multivariate_normal(mean, cov, size=n)
    return np.array(samples)



def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    mu_n = mu + (x-mu)/n
    return mu_n


def _plot_sequence_estimate():
    data = gen_data(100,2, np.array([0,0]), 3) # Set this as the data
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        next_mu = update_sequence_mean(estimates[i], data[i], i+1)
        estimates.append(next_mu)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label = 'Second dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    y_hat = np.array(y_hat)
    error = (y-y_hat)**2
    return np.mean(error,axis = 1)


def _plot_mean_square_error():
    data = gen_data(100,2, np.array([0,0]), 3) # Set this as the data
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        next_mu = update_sequence_mean(estimates[i], data[i], i+1)
        estimates.append(next_mu)
        
    sqerror = _square_error(np.array([0,0]), estimates)
    plt.plot(sqerror)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    # remove this if you don't go for the independent section
    cov = (sigma**2) * np.identity(k)  # Diagonal covariance matrix
    
    # Precompute a linear adjustment to the mean
    delta_mean = (end_mean - start_mean) / (n - 1)  # Step size for the mean change
    
    samples = []
    for i in range(n):
        # Interpolate the mean for the current step
        current_mean = start_mean + i * delta_mean
        # Generate a sample with the current mean
        sample = np.random.multivariate_normal(current_mean, cov)
        samples.append(sample)
    
    return np.array(samples)

def update_sequence_mean_changing(
    mu: np.ndarray,
    x: np.ndarray,
    alpha: float
) -> np.ndarray:
    '''Performs the mean sequence estimation update with a forgetting factor (alpha)'''
    # Update the mean using the forgetting factor
    mu_n = alpha * x + (1 - alpha) * mu
    return mu_n

def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    n = 500  # Number of samples
    k = 3    # Number of dimensions
    start_mean = np.array([0, 1, -1])  # Starting mean vector
    end_mean = np.array([1, -1, 0])    # Ending mean vector
    sigma = 3.0  # Standard deviation for each dimension
    alpha = 0.1 # forgetting factor
    data = gen_changing_data(n, k, start_mean, end_mean, sigma) # Set this as the data
    estimates = [np.zeros(k)]
    for i in range(data.shape[0]):
        next_mu = update_sequence_mean_changing(estimates[i], data[i],alpha)
        estimates.append(next_mu)
    for i in range(k):
        plt.plot([e[i] for e in estimates], label=f'Dimension {i+1}')
    plt.legend(loc='upper center')
    plt.show()


#if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
#    pass
