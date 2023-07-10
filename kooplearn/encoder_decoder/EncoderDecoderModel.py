from kooplearn.data.utils.TimeseriesDataModule import TimeseriesDataModule

class EncoderDecoderModel:
    def __init__(self, feature_map, koopman_estimator, decoder):
        self.feature_map = feature_map
        self.koopman_estimator = koopman_estimator
        self.decoder = decoder
        self.datamodule = None
        self.initialize_model()

    def initialize_model(self):
        self.feature_map.initialize(self)
        self.koopman_estimator.initialize(self)
        self.decoder.initialize(self)

    def fit(self, X, Y, datamodule_kwargs=None):
        if datamodule_kwargs is not None:
            self.datamodule = TimeseriesDataModule(**datamodule_kwargs)
        self.feature_map.fit(X, Y, datamodule_kwargs)
        self.koopman_estimator.fit(X, Y, datamodule_kwargs)
        self.decoder.fit(X, Y, datamodule_kwargs)

    def predict(self, X):
        return self.decoder(self.koopman_estimator.predict(self.feature_map(X)))
