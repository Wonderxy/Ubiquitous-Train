import numpy as np

def fit(tensor1,tensor2):
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
    return 1-(np.linalg.norm(tensor1-tensor2)/np.linalg.norm(tensor2))