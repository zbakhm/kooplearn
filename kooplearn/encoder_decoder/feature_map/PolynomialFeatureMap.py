from sklearn.preprocessing import PolynomialFeatures
from numpy.typing import ArrayLike

from kooplearn.encoder_decoder.decoder.CustomDecoder import CustomDecoder
from kooplearn.encoder_decoder.feature_map.FeatureMap import FeatureMap


class PolynomialFeatureMap(FeatureMap):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)

    def __call__(self, X: ArrayLike):
        return self.poly.fit_transform(X)

    @staticmethod
    def decoder_from_feature_map(num_features: int):
        def decoder_fn(x):
            return x[:, 1:num_features+1]
        return CustomDecoder(decoder_fn)
