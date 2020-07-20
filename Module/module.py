from typing import Iterator

import inspect

from Tensor.parameter import Parameter

class Module(object):
    """
        Abstract class to define all the module

    """

    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, *inputs):
        raise NotImplementedError
