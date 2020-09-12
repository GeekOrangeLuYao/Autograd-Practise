from typing import Dict, Callable
import numpy as np

from Tensor.tensor import Tensor


class Layer(object):

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = dict()
        self.grads: Dict[str, Tensor] = dict()

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size: int, output_size: int) -> None:
        super(Linear, self).__init__()
        self.params["weight"] = np.random.randn(input_size, output_size)
        self.params["bias"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["weight"] + self.params["bias"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["bias"] = np.sum(grad, axis=0)
        # TODO The Tensor.T should be implemented
        self.grads["weight"] = self.inputs.T @ grad
        return grad @ self.params["weight"].T
