import tensorly.backend as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from utils.forList import factorial_list

def index_t2v(tensorShp,indexList):
    """Obtain the index of tensor form corresponding to vector form

    Parameters
    ----------
    tensorShp : tensor shape
    indexList : tuple/list, Index in Tensor Form

    Returns
    -------
    index : int, Index in vector form
    """
    index = 0
    shp = tensorShp
    i = 0
    while i < len(indexList):
        if i < len(indexList)-1:
            index += factorial_list(shp[i+1:])*indexList[i]
        elif i == len(indexList)-1:
            index += indexList[i]
        i += 1
    return index

def index_v2t(tensorShp,index):
    """Obtain the index of vector form corresponding to tensor form

    Parameters
    ----------
    tensorShp : tensor shape
    index : int, Index in Vector Form

    Returns
    -------
    indexList : list, Index in Tensor Form
    """
    indexList = []
    shp = tensorShp
    for i in range(len(shp)):
        j = int(index/factorial_list(shp[i+1:]))
        indexList.append(j)
        index -= j*factorial_list(shp[i+1:])

    return indexList


def tensor_to_vec(tensor):
    """Vectorises a tensor

    Parameters
    ----------
    tensor : ndarray
             tensor of shape ``(i_1, ..., i_n)``

    Returns
    -------
    1D-array
        vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
    """
    return tl.reshape(tensor, (-1,))


def vec_to_tensor(vec, shape):
    """Folds a vectorised tensor back into a tensor of shape `shape`

    Parameters
    ----------
    vec : 1D-array
        vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
    shape : tuple
        shape of the ful tensor

    Returns
    -------
    ndarray
        tensor of shape `shape` = ``(i_1, ..., i_n)``
    """
    return tl.reshape(vec, shape)


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`
        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return tl.moveaxis(tl.reshape(unfolded_tensor, full_shape), 0, mode)
