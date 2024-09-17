import tensorly as tl
import sys
import numpy as np
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tt.tt_tensor import validate_tt_rank, TTTensor
from tensor.svd import svd_interface


def tensor_train(input_tensor, rank, svd="truncated_svd", verbose=False):
    """TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor/ndarray
    rank : {int, int list}
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
              order-3 tensors of the TT decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method=svd)

        rank[k + 1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if verbose is True:
            print(
                "TT factor " + str(k) + " computed with shape " + str(factors[k].shape)
            )

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    if verbose is True:
        print(
            "TT factor "
            + str(n_dim - 1)
            + " computed with shape "
            + str(factors[n_dim - 1].shape)
        )

    return TTTensor(factors)

def truncated(sv, delta):
    """
    Used in A, truncating the singular value matrix based on delta

    Parameters
    ----------
    sv : ndarry, matrix
    delta : float

    Returns
    -------
    r : ttd_rank

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    sv = sv[::-1] #逆序
    sv = np.power(sv, 2)
    sv = np.cumsum(sv)
    r = sv.size
    for i in range(sv.size):
        if sv[i] <= np.power(delta, 2):
            r -= 1
    return r


def tt_svd(input_tensor, rank, svd="truncated_svd", verbose=False):
    """TT decomposition via recursive SVD

    Parameters
    ----------
    input_tensor : tensorly.tensor/ndarray
    rank : float, decomposition accuracy/eps
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
              order-3 tensors of the TT decomposition
    """
    shp = np.array(input_tensor.shape)
    L = len(shp)
    R = [1 for x in range(L+1)]
    G = []
    C = input_tensor

    for i in range(1,L):
        row = R[i-1] * shp[i-1]
        col = int(C.size / row)
        C = tl.reshape(C,(row, col))
        U,sigma,VT = np.linalg.svd(C,full_matrices=False)

        delta = rank/(np.sqrt(L-1))
        # R[i] = truncated(U,sigma,VT,C,delta)
        R[i] = truncated(sigma,delta*tl.norm(C))
        # print("R{}=".format(i),R[i])

        U = U[:,:R[i]]
        sigma = np.diag(sigma[:R[i]])
        VT = VT[:R[i],:]

        G.append(U.reshape((R[i-1],shp[i-1],R[i])))
        C = np.dot(sigma, VT)

    G.append(C.reshape(R[L-1],shp[L-1],R[L]))
    return TTTensor(G)

TT_FUNS = ["tensor_train","tt_svd"]

class TensorTrain(DecompositionMixin):
    """Decompose a tensor into a matrix in tt-format

    Parameters
    ----------
    tensor : tensorized matrix
        if your input matrix is of size (4, 9) and your tensorized_shape (2, 2, 3, 3)
        then tensor should be tl.reshape(matrix, (2, 2, 3, 3))
    rank : 'same', float or int tuple
        - if 'same' creates a decomposition with the same number of parameters as `tensor`
        - if float, creates a decomposition with `rank` x the number of parameters of `tensor`
        - otherwise, the actual rank to be used, e.g. (1, rank_2, ..., 1) of size tensor.ndim//2. Note that boundary conditions dictate that the first rank = last rank = 1.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    tt_matrix
    """

    def __init__(self, rank=1.0, method="tt_svd", svd="truncated_svd", verbose=False):
        self.rank = rank
        self.method = method
        self.svd = svd
        self.verbose = verbose

    def fit_transform(self, tensor):
        '''
        fit and transform
        '''
        if self.method == "tensor_train":
            tt_fun = tensor_train
        elif self.method == "tt_svd":
            tt_fun = tt_svd
        else:
            raise ValueError(
                f"Got method={self.method}. However, the possible choices are {TT_FUNS} or to pass a callable."
            )
        
        self.decomposition_ = tt_fun(
            tensor, rank=self.rank, svd=self.svd, verbose=self.verbose
        )
        return self.decomposition_
    

if __name__ == "__main__":
    from experiments.load_data import load_tensor
    A = load_tensor(["ratingTensor"])[0]
    tt = TensorTrain(rank=0,method="tt_svd").fit_transform(A) 
    print(type(tt))
    print(tt.rank)
    print(tt.shape)
    print(tt[0].shape)
