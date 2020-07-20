import numpy as np

from Tensor.tensor import Tensor

class Parameter(Tensor):

    def __init__(self, *shape) -> None:
        super(Parameter, self).__init__(np.random.rand(*shape), requires_grad=True)
