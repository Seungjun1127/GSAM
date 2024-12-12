import numpy as np
from typing import Union
from numpy.linalg import inv
from scipy.linalg import cholesky, LinAlgError

def compute_gsam(embeddings: np.ndarray,
                 metric: str = 'kl_divergence') -> float:
    """
    Compute the Gaussian Semantic Alignment Metric (GSAM) for a given set of embeddings
    without any dimensionality reduction. Directly uses the provided high-dimensional embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        A 2D array of shape (N, D) representing N samples of D-dimensional embeddings.
    metric : str
        Metric for Gaussian alignment: 'kl_divergence' or 'neg_log_likelihood'.

    Returns
    -------
    float
        The GSAM score. Higher indicates closer alignment to a Gaussian distribution.
    """
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
        kl_val = compute_kl_divergence_from_gaussian(emb_normalized, mu, sigma)
        gsam_score = -kl_val
    elif metric == 'neg_log_likelihood':
        nll_val = compute_neg_log_likelihood(emb_normalized, mu, sigma)
        gsam_score = -nll_val
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return gsam_score


def batch_compute_gsam(batch_embeddings: Union[list, np.ndarray],
                       metric: str = 'kl_divergence') -> float:
    """
    Compute GSAM for multiple batches of embeddings and return the average score.

    Parameters
    ----------
    batch_embeddings : list or np.ndarray
        If np.ndarray, considered as a single batch. If list of np.ndarray, multiple batches.
    metric : str
        GSAM metric: 'kl_divergence' or 'neg_log_likelihood'

    Returns
    -------
    float
        Average GSAM score across batches.
    """
    if isinstance(batch_embeddings, np.ndarray):
        batch_embeddings = [batch_embeddings]

    scores = []
    for emb in batch_embeddings:
        score = compute_gsam(emb, metric=metric)
        scores.append(score)
    return np.mean(scores)


def compare_models(model_embeddings: dict, metric: str = 'kl_divergence') -> dict:
    """
    Compare multiple models' embeddings using GSAM.

    Parameters
    ----------
    model_embeddings : dict
        Dictionary where keys are model names and values are embeddings (np.ndarray).
    metric : str
        'kl_divergence' or 'neg_log_likelihood'

    Returns
    -------
    dict
        {model_name: GSAM_score}
    """
    results = {}
    for model_name, emb in model_embeddings.items():
        score = compute_gsam(emb, metric=metric)
        results[model_name] = score
    return results


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to have zero mean and unit variance per dimension.
    Handle zero variance dimensions by leaving them as is (std=1 if std=0).
    """
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    std[std == 0] = 1.0
    return (embeddings - mean) / std


def estimate_gaussian_params(embeddings: np.ndarray):
    """
    Estimate mean and covariance matrix of the embeddings.
    """
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def regularize_covariance(sigma: np.ndarray, epsilon: float = 1e-6, max_retries: int = 10) -> np.ndarray:
    """
    Regularize the covariance matrix to ensure it is positive-definite and invertible.
    We try Cholesky decomposition. If it fails, we add epsilon*I and retry.

    Parameters
    ----------
    sigma : np.ndarray
        Covariance matrix.
    epsilon : float
        Small value to add to the diagonal if needed.
    max_retries : int
        Maximum number of times to try increasing epsilon.

    Returns
    -------
    np.ndarray
        A regularized covariance matrix that is positive-definite.
    """
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
    """
    Compute log probability density of x under a multivariate Gaussian N(mu, sigma).

    log p(x) = -0.5 * (D * log(2π) + log|Σ| + (x - μ)^T Σ^-1 (x - μ))
    """
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


def compute_kl_divergence_from_gaussian(embeddings: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute KL divergence D(P_data || P_gaussian).
    P_data is discrete uniform over N samples.
    
    KL(P_data||P_gaussian) = (1/N)*∑ log(P_data(x_i)/P_gaussian(x_i))
                           = (1/N)*∑ [log(1/N) - log p_gaussian(x_i)]
                           = -log(N) - (1/N)*∑ log p_gaussian(x_i)
    """
    N = embeddings.shape[0]
    log_probs = [log_multivariate_gaussian_pdf(x, mu, sigma) for x in embeddings]
    avg_logp = np.mean(log_probs)
    kl = -np.log(N) - avg_logp
    return kl


def compute_neg_log_likelihood(embeddings: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute negative log-likelihood under the fitted Gaussian.
    NLL = - (1/N)*∑ log p_gaussian(x_i)
    """
    log_probs = [log_multivariate_gaussian_pdf(x, mu, sigma) for x in embeddings]
    avg_logp = np.mean(log_probs)
    nll = -avg_logp
    return nll
