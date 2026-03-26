import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    unique_labels = np.unique(predictions)
    counts = np.array([np.sum(predictions == label, axis=0) for label in unique_labels])
    best_label_indices = np.argmax(counts, axis=0)
    return list(unique_labels[best_label_indices])
        