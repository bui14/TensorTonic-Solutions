def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)

    N = len(actual_tokens)
    target_probs = prob_distributions[np.arange(N), actual_tokens]

    target_probs = np.clip(target_probs, 1e-15, 1.0)

    H = -np.mean(np.log(target_probs))

    return np.exp(H)