import tensorly as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tt.tt_tensor import validate_tt_rank, TTTensor
from tenalg.svd import svd_interface


def tensor_train(input_tensor, rank, svd="truncated_svd", verbose=False):
    """TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
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

    def __init__(self, rank, svd="truncated_svd", verbose=False):
        self.rank = rank
        self.svd = svd
        self.verbose = verbose

    def fit_transform(self, tensor):
        '''
        fit and transform
        '''
        self.decomposition_ = tensor_train(
            tensor, rank=self.rank, svd=self.svd, verbose=self.verbose
        )
        return self.decomposition_