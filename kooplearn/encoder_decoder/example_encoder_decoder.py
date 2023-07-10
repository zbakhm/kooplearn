from kooplearn.encoder_decoder.EncoderDecoderModel import EncoderDecoderModel
from kooplearn.encoder_decoder.feature_map.PolynomialFeatureMap import PolynomialFeatureMap
from kooplearn.encoder_decoder.koopman_estimators.ExtendedDMD import ExtendedDMD
import numpy as np

num_features = 10
num_samples = 100
data = np.random.rand(num_samples, num_features)
X = data[:-1]
Y = data[1:]
feature_map = PolynomialFeatureMap(degree=6)
decoder = feature_map.decoder_from_feature_map(num_features=num_features)
koopman_estimator = ExtendedDMD(feature_map=feature_map, rank=5)
model = EncoderDecoderModel(feature_map=feature_map,
                            koopman_estimator=koopman_estimator,
                            decoder=decoder)
model.fit(X, Y)





