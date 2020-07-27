from typing import List, NamedTuple, Callable, Union, Optional

import numpy as np
from Tensor.tensor_ops import *

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arr: Arrayable) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return np.array(arr)

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensor: Tensorable) -> 'Tensor':
    if isinstance(tensor, Tensor):
        return tensor
    else:
        return Tensor(tensor)


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor(object):

    def __init__(self,
                 data: Arrayable,
                 requires_grad = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = data.shape
        self.grad: Optional['Tensor'] = None # Note grad.requires_grad should be False

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad = {self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, f"The required_grad is {self.requires_grad}"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError(f"grad of {id(self)} must be specified for non-0-tensor")

        self.grad.data += grad.data # need to write __iadd__

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    def __add__(self, other) -> 'Tensor':
        return tensor_add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        return tensor_add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        self.data = self.data + ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        return tensor_mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return tensor_mul(ensure_tensor(other), self)

    def __imul__(self, other) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data
        return self

    def __matmul__(self, other) -> 'Tensor':
        return tensor_matmul(self, ensure_tensor(other))

    def __neg__(self) -> 'Tensor':
        return tensor_neg(self)

    def __sub__(self, other) -> 'Tensor':
        return tensor_add(self, -ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return tensor_add(-self, ensure_tensor(other))

    def __isub__(self, other) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        return self

    def __getitem__(self, item) -> 'Tensor':
        return tensor_slice(self, item)

    @property
    def T(self) -> 'Tensor':
        return tensor_transpose(self)