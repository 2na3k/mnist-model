import torch

from torch import Tensor

from typing import Tuple

def permute_data(x: Tensor, y: Tensor, seed=1) -> Tuple[Tensor]:
    perm = torch.randperm(x.shape[0])
    return x[perm], y[perm]

def assert_dim(t: Tensor, dim: Tensor):
    ''' 
    Suppose to ensure the dimension of both are the same
    '''
    assert len(t.shape) == dim, 'tensor expected to have dimension {0}, instead of {1}'.format(dim, len(t.shape))

