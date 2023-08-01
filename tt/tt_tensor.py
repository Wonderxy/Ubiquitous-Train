import tensorly as tl
import numpy as np
from tensorly._factorized_tensor import FactorizedTensor

def _validate_tt_tensor(tt_tensor):
    factors = tt_tensor
    n_factors = len(factors)

    if isinstance(tt_tensor, TTTensor):
        # it's already been validated at creation
        return tt_tensor.shape, tt_tensor.rank
    elif isinstance(tt_tensor, (float, int)):  # 0-order tensor
        return 0, 0

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank, current_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor) == 3:
            raise ValueError(
                "TT expresses a tensor as third order factors (tt-cores).\n"
                f"However, tl.ndim(factors[{index}]) = {tl.ndim(factor)}"
            )
        # Consecutive factors should have matching ranks
        if index and tl.shape(factors[index - 1])[2] != current_rank:
            raise ValueError(
                "Consecutive factors should have matching ranks\n"
                " -- e.g. tl.shape(factors[0])[2]) == tl.shape(factors[1])[0])\n"
                f"However, tl.shape(factor[{index-1}])[2] == {tl.shape(factors[index - 1])[2]} but"
                f" tl.shape(factor[{index}])[0] == {current_rank} "
            )
        # Check for boundary conditions
        if (index == 0) and current_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[0].shape[0] == 1."
                f"However, got factor[0].shape[0] = {current_rank}."
            )
        if (index == n_factors - 1) and next_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[-1].shape[2] == 1."
                f"However, got factor[{n_factors}].shape[2] = {next_rank}."
            )

        shape.append(current_shape)
        rank.append(current_rank)

    # Add last rank (boundary condition)
    rank.append(next_rank)

    return tuple(shape), tuple(rank)


def tt_to_tensor(factors):
    """Returns the full tensor whose TT decomposition is given by 'factors'
        Re-assembles 'factors', which represent a tensor in TT/Matrix-Product-State format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TT factors (TT-cores)
    Returns
    -------
    output_tensor : ndarray
                   tensor whose TT/MPS decomposition was given by 'factors'
    """
    if isinstance(factors, (float, int)):  # 0-order tensor
        return factors

    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)


def tt_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in TT (or Tensor-Train) format
    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors
    mode: int
          unfolding matrix to be computed along this mode
    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(tt_to_tensor(factors), mode)


def tt_to_vec(factors):
    """Returns the tensor defined by its TT format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors
    Returns
    -------
    1-D array
    vectorized format of tensor defined by 'factors'
    """
    return tl.tensor_to_vec(tt_to_tensor(factors))



class TTTensor(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()

        # Will raise an error if invalid
        shape, rank = _validate_tt_tensor(factors)

        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    def __getitem__(self, index):
        return self.factors[index]

    def __setitem__(self, index, value):
        self.factors[index] = value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        message = f"factors list : rank-{self.rank} matrix-product-state tensor of shape {self.shape} "
        return message

    def to_tensor(self):
        return tt_to_tensor(self)

    def to_unfolding(self, mode):
        return tt_to_unfolded(self, mode)

    def to_vec(self):
        return tt_to_vec(self)
    
    