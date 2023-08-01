import tensorly.backend as tl


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
