from kooplearn.data.utils.TimeseriesDataModule import TimeseriesDataModule


class DPNetsModel:
    def __init__(self, feature_map, koopman_estimator):
        self.feature_map = feature_map
        self.koopman_estimator = koopman_estimator
        self.datamodule = None
        self.initialize_model()

    def initialize_model(self):
        self.feature_map.initialize()

    def fit(self, X, Y, datamodule_kwargs=None):
        if datamodule_kwargs is not None:
            self.datamodule = TimeseriesDataModule(**datamodule_kwargs)
        if not self.feature_map.is_fitted:
            self.feature_map.fit(X, Y, datamodule_kwargs)
        if X is None or Y is None:
            X, Y = self.feature_map.datamodule.train_dataset.get_X_Y_numpy_matrices()
        X = self.feature_map(X)
        Y = self.feature_map(X)
        self.koopman_estimator.fit(X, Y)

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        X = self.feature_map(X)
        Y = self.koopman_estimator.predict(X, t, observables)
        return Y
