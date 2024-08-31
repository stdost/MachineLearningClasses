# Author: Stella Stoyanova
# Date:
# Project: 
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    features = []
    target_class = []
    for i, (loc, scale) in enumerate(zip(locs, scales)):
        # Generate n samples from the normal distribution with the given loc (mean) and scale (SD)
        r = norm.rvs(loc=loc, scale=scale, size=n)
        features.append(r)
        target_class.append(np.full(n, i))  #assign same class label to all generated samples
    features = np.concatenate(features)
    target_class = np.concatenate(target_class)
    classes = np.arange(len(locs))

    # return all 3
    return features, target_class, classes



def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    class_features = features[targets == selected_class]

    # If no samples belong to the selected class
    if class_features.shape[0] == 0:
        return np.nan  # missing data
    
    # Compute the mean of the selected class
    return np.mean(class_features, axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    class_features = features[targets == selected_class]

    # If no samples belong to the selected class
    if class_features.shape[0] < 2:
        return np.nan  # missing data
    
    # Compute the mean of the selected class
    return np.cov(class_features, rowvar=False)



def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    if feature.ndim == 1:
        # Univariate normal distribution
        return norm.pdf(feature, loc=class_mean, scale=np.sqrt(class_covar))
    
    # Multidimensional case (multivariate normal distribution)
    return multivariate_normal.pdf(feature, mean=class_mean, cov=class_covar)



def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        class_mean = mean_of_class(train_features, train_targets, class_label)
        means.append(class_mean)
        class_cov = covar_of_class(train_features, train_targets, class_label)
        covs.append(class_cov)
    likelihoods = np.zeros((test_features.shape[0], len(classes)))
    for i in range(test_features.shape[0]):
        for j, (mean, cov) in enumerate(zip(means, covs)):
            likelihoods[i,j] = likelihood_of_class(test_features[i], mean, cov)
    return np.array(likelihoods)



def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis =1)


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """

