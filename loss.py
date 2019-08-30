import numpy as np
from tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor,actual: Tensor) -> Tensor:
        raise NotImplementedError

#total squared error
class MSE(Loss):
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        return np.square((predicted-actual)**2).mean()

    def grad(self, predicted: Tensor,actual: Tensor) -> Tensor:
        return 2*(predicted-actual)
