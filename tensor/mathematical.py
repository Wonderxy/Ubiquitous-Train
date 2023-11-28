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


def factors_kron(factorA, factorB):
    """Kronecker product 3-order tensor(factor)

    Parameters
    ----------
    factorA : ndarray/tensor
    factorB : ndarray/tensor

    Returns
    -------
    tensor(3-order)
        The result of Two Tensors' Kronecker product
    """
    shpA = factorA.shape
    shpB = factorB.shape
    factors = np.zeros((shpA[0]*shpB[0], shpA[1], shpA[2]*shpB[2]))#8.16
    for i in range(factorA.shape[1]):
        factors[:,i,:] = np.kron(factorA[:,i,:], factorB[:,i,:])
    return factors
