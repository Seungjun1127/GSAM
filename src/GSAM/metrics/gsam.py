import numpy as np
from typing import Union
from numpy.linalg import inv
from scipy.linalg import cholesky, LinAlgError

def compute_gsam(embeddings: np.ndarray,
                 metric: str = 'kl_divergence') -> float:

    # Basic validations
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array (N, D).")
    N, D = embeddings.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to estimate Gaussian parameters.")
    # If variance is zero in any dimension, normalization handles it safely.

    emb_normalized = normalize_embeddings(embeddings)
    mu, sigma = estimate_gaussian_params(emb_normalized)
    sigma = regularize_covariance(sigma)

    if metric == 'kl_divergence':
        kl_val = compute_kl_divergence(emb_normalized, mu, sigma)
        gsam_score = -kl_val
    elif metric == 'neg_log_likelihood':
        nll_val = compute_neg_log_likelihood(emb_normalized, mu, sigma)
        gsam_score = -nll_val
    elif metric == 'chi_squared':
        chi_squared_val = chi_squared_goodness_of_fit(emb_normalized, mu, sigma)
        gsam_score = chi_squared_val
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return gsam_score


def batch_compute_gsam(batch_embeddings: Union[list, np.ndarray],
                       metric: str = 'kl_divergence') -> float:

    if isinstance(batch_embeddings, np.ndarray):
        batch_embeddings = [batch_embeddings]

    scores = []
    for emb in batch_embeddings:
        score = compute_gsam(emb, metric=metric)
        scores.append(score)
    return np.mean(scores)


def compare_models(model_embeddings: dict, metric: str = 'kl_divergence') -> dict:

    results = {}
    for model_name, emb in model_embeddings.items():
        score = compute_gsam(emb, metric=metric)
        results[model_name] = score
    return results


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:

    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    std[std == 0] = 1.0
    return (embeddings - mean) / std


def estimate_gaussian_params(embeddings: np.ndarray):

    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def regularize_covariance(sigma: np.ndarray, epsilon: float = 1e-6, max_retries: int = 10) -> np.ndarray:

    attempt = 0
    while attempt < max_retries:
        try:
            _ = cholesky(sigma)
            return sigma  # success
        except LinAlgError:
            sigma = sigma + np.eye(sigma.shape[0]) * epsilon
            epsilon *= 10  # increment epsilon exponentially for next try
            attempt += 1

    # If still not PD after retries, raise error or return last attempt
    # In practice, raising might be better to debug data issues.
    return sigma


def log_multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:

    D = mu.shape[0]
    sign, logdet = np.linalg.slogdet(sigma)
    # We assume sigma is PD, so sign should be > 0.
    # For safety, if sign <= 0, handle gracefully:
    if sign <= 0:
        # fallback: add a bit to diagonal and try again.
        sigma = regularize_covariance(sigma)
        sign, logdet = np.linalg.slogdet(sigma)

    inv_sigma = inv(sigma)
    diff = x - mu
    return -0.5 * (D * np.log(2*np.pi) + logdet + diff @ inv_sigma @ diff)


def compute_neg_log_likelihood(embeddings: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    log_probs = [log_multivariate_gaussian_pdf(x, mu, sigma) for x in embeddings]
    
    # Calculate the maximum log probability density
    max_log_prob = np.max(log_probs)
    
    # Calculate the average log probability density
    avg_logp = np.mean(log_probs)
    
    # Calculate the normalized NLL score
    nll = -avg_logp
    normalized_score = 100 - (avg_logp - max_log_prob) / max_log_prob * 100  # Convert to percentage
    return normalized_score


def compute_kl_divergence(embeddings: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Calculate the Kullback-Leibler divergence between the data distribution and a Gaussian distribution."""
    N = embeddings.shape[0]
    log_probs = [log_multivariate_gaussian_pdf(x, mu, sigma) for x in embeddings]
    avg_logp = np.mean(log_probs)
    kl_divergence = -avg_logp  # KL divergence is the negative average log probability
    return kl_divergence


def chi_squared_goodness_of_fit(embeddings: np.ndarray, mu: np.ndarray, sigma: np.ndarray, bins: int = 30, max_chi_squared: float = 100.0) -> float:
    # Z-score normalization
    normalized_embeddings = (embeddings - mu) / sigma
    
    # Create histogram of the normalized data
    hist, bin_edges = np.histogram(normalized_embeddings, bins=bins, density=True)
    
    # Calculate the expected Gaussian distribution for normalized data
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    expected = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (bin_centers ** 2))
    
    # Calculate Chi-Squared statistic
    chi_squared = np.sum((hist - expected) ** 2 / expected)
    
    # Normalize the Chi-Squared statistic to a range of 0 to 100
    normalized_score = max(0, (1 - chi_squared / max_chi_squared) * 100)
    
    return normalized_score
