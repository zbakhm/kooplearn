from numpy.typing import ArrayLike
from kooplearn.encoder_decoder.DPNetsModel import EncoderDecoderModel


class FeatureMap:
    def __init__(self):
        is_fitted = False

    def __call__(self, X: ArrayLike):
        pass

    def initialize(self, model: EncoderDecoderModel):
        pass

    def fit(self, X, Y):
        pass
