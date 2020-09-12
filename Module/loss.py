import numpy as np
from Module.module import Module
from Tensor.tensor import Tensor


class Loss(object):

    def loss(self,
             predicted: Tensor,
             actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self,
             predicted: Tensor,
             actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    def loss(self,
             predicted: Tensor,
             actual: Tensor) -> float:
        return np.mean((predicted - actual) * (predicted - actual))  # TODO: Use __pow__

    def grad(self,
             predicted: Tensor,
             actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
