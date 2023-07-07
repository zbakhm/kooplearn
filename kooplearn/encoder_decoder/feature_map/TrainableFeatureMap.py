import abc
from typing import Optional

from kooplearn.encoder_decoder.feature_map.FeatureMap import FeatureMap
from numpy.typing import ArrayLike


class TrainableFeatureMap(FeatureMap):
    #Trainable feature maps should accept numpy arrays and return numpy arrays. Internally thay can do whatever.
    @abc.abstractmethod
    def fit(self, X: Optional[ArrayLike], Y: Optional[ArrayLike]):
        pass