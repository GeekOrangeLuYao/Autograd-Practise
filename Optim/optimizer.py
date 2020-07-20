
from Tensor.parameter import Parameter

class Optimizer(object):
    """
        Abstract optimizer
    """

    def step(self):
        raise NotImplementedError



class SGD(Optimizer):

    def __init__(self):
        raise NotImplementedError