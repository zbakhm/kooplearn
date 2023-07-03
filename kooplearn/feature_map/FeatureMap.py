import abc
from typing import Optional

from numpy.typing import ArrayLike


class FeatureMap(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: ArrayLike):
        pass

    def cov(self, X: ArrayLike, Y: Optional[ArrayLike] = None):
        phi_X = self.__call__(X)
        if Y is None:
            c = phi_X.T@phi_X
        else:
            phi_Y = self.__call__(Y)
            c = phi_X.T@phi_Y
        c *= (X.shape[0])**(-1)
        return c
