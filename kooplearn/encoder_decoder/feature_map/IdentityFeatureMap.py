from kooplearn.encoder_decoder.feature_map.FeatureMap import FeatureMap
from numpy.typing import ArrayLike

class IdentityFeatureMap(FeatureMap):
    def __call__(self, X: ArrayLike):
        return X