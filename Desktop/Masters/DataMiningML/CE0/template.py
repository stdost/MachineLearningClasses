import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    epsilon = np.finfo(float).eps
    sigma = np.maximum(sigma, epsilon)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    xs = np.linspace(x_start, x_end, num=500)
    plt.plot(xs, normal(xs, sigma, mu))

def _plot_three_normals():
    # Part 1.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    #plt.savefig('1_2_1.png')
    plt.show()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    mixture_d = np.zeros_like(x, dtype=np.float64)
    for sigma, mu, weight in zip(sigmas,mus,weights):
        mixture_d += weight * normal(x,sigma, mu)
    return mixture_d

def plot_mixture(sigmas, mus, weights, x_start, x_end):
    xs = np.linspace(x_start, x_end, num=500)
    mixture_d = normal_mixture(xs, sigmas, mus, weights)
    plt.plot(xs, mixture_d)

def _compare_components_and_mixture():
    # Part 2.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1.5, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_mixture([0.5,1.5, 0.25],[0,-0.5,-1.5], [1/3, 1/3, 1/3], -5, 5 )
    #plt.savefig('2_2_1.png')
    plt.show()


def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    # Determine how many samples to draw from each component
    n_samples_per_component = np.random.multinomial(n_samples, weights)
    
    samples = []
    
    # Generate samples for each Gaussian component
    for sigma, mu, n in zip(sigmas, mus, n_samples_per_component):
        component_samples = np.random.normal(mu, sigma, n)
        samples.append(component_samples)
    
    # Combine all samples into a single array
    samples = np.concatenate(samples)
    
    return samples

def plot_hist(sigmas, mus, weights, n_samples):
    samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples)
    plt.hist(samples, 100, density=True)

def _plot_mixture_and_samples():
    # Part 3.2
    sample_sizes = [10, 100, 500, 1000]
    sigmas = [0.3, 0.5, 1]
    mus = [0, -1, 1.5]
    weights = [0.2, 0.3, 0.5]
    
    plt.figure(figsize=(16, 4))
    
    for i, n_samples in enumerate(sample_sizes):
        plt.subplot(1, 4, i+1)  # Select subplot 1x4 grid, position i+1
        plot_hist(sigmas, mus, weights, n_samples)
        plot_mixture(sigmas, mus, weights, -10, 10)
        plt.title(f'Sample Size: {n_samples}')
    
    plt.tight_layout()
    plt.savefig('3_2_1.png')
    plt.show()

#if __name__ == '__main__':
    # select your function to test here and do `python3 template.py'