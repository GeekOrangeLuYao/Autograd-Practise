from typing import Callable
import numpy as np

from Tensor.tensor import Tensor
from Module.layer import Layer

Activation_Function = Callable[[Tensor], Tensor]

class Activation(Layer):

    def __init__(self,
                 f: Activation_Function,
                 f_prime: Activation_Function) -> None:
        super(Activation, self).__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor):
        # TODO
        raise NotImplementedError