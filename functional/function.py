"""
    Sigmoid: 1 / (1 + exp(x))
    Tanh: tanh(x)
    ReLU: max(0, x)
    Leaky ReLU: leaky_relu(x) = max(0.01x, x)
    ELU: ELU(x) = x if x > 0 else alpha *(exp(x) -1) # alpha
    SELU: something new

"""

import numpy as np

from Tensor.tensor import Tensor, Dependency


def tanh(tensor: Tensor) -> Tensor:
    """
        tanh(s) = (exp(s) - exp(-s)) / (exp(s) + exp(-s))
        tanh'(s) = 1 - tanh(s) tanh(s)
    """
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    """
        sigmoid(s) = 1. / (1. + exp(s))
        sigmoid'(s) = sigmoid(s) * (1. - sigmoid(s))
    """
    data = 1. / (1. + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1. - data)
        depends_on = [Dependency(tensor, grad_fn)]
    else :
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def relu(tensor: Tensor) -> Tensor:
    """
        relu(s) = max(0., s)
        relu'(s) = 1 if s >= 0 else 0
    """
    data = np.maximum(tensor.data, 0)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.array(data >= 0.0, dtype=np.int)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)