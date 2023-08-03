import numpy as np

def fit(tensorA,tensorB):
    """Calculating the Fit of Two Tensors

    Parameters
    ----------
    tensorA : ndarray/list
    tensorB : ndarray/list

    Returns
    -------
    float
        The Fit of Two Tensors
    """
    return 1-(np.linalg.norm(tensorA-tensorB)/np.linalg.norm(tensorB))