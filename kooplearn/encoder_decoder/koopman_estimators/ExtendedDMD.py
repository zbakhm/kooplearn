import numpy as np

from kooplearn.encoder_decoder.koopman_estimators.KoopmanEstimator import KoopmanEstimator


def dmd_algorithm(X, Y, rank, method='classical'):
    # DMD algorithm to approximate A = Y @ X^+
    if method == 'classical':
        # SVD of X_encoded
        left_singular_vectors, singular_values, right_singular_values_hermitian = np.linalg.svd(X)
        # truncate SVD
        singular_values = singular_values[:rank]
        left_singular_vectors = left_singular_vectors[:, :rank]
        right_singular_values_hermitian = right_singular_values_hermitian[:rank, :]
        A_tilde = left_singular_vectors.conj().T @ Y @ right_singular_values_hermitian.conj().T @ np.diag(
            1 / singular_values)
        # eigendecomposition of A_tilde (r projection of A)
        eigenvalues, eigenvectors_A_tilde = np.linalg.eig(A_tilde)
        # DMD modes (eigenvectors of A)
        modes = Y @ right_singular_values_hermitian.conj().T @ np.diag(1 / singular_values) \
                @ eigenvectors_A_tilde
        A = modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes)
    elif method == 'naive':
        # Usually we do not want to explicitly compute A, but what if we do?
        A, residuals, rank, singular_values = np.linalg.lstsq(X, Y, rcond=None)
        eigenvalues, modes = np.linalg.eig(A)
    else:
        raise ValueError('method must be one of \'classical\' or \'naive\'')
    return modes, eigenvalues, A


class ExtendedDMD(KoopmanEstimator):
    def __init__(self, feature_map, rank, sample_rate=1.0):
        super().__init__()
        self.rank = rank
        self.feature_map = feature_map
        self.sample_rate = sample_rate
        self.modes_ = None
        self.eigenvalues_ = None
        self.koopman_operator_ = None
        self.initial_condition_ = None

    def fit(self, X, Y):
        X_encoded = self.feature_map(X)
        Y_encoded = self.feature_map(Y)
        self.modes_, self.eigenvalues_, self.koopman_operator_ = dmd_algorithm(X_encoded, Y_encoded, self.rank)
        x1 = X_encoded[:, 0]
        # initial condition b = pinv(modes) @ x1
        self.initial_condition_ = np.linalg.lstsq(self.modes_, x1, rcond=None)[0]

    def predict(self, X, step=1):
        X_encoded = self.feature_map(X)
        return np.linalg.matrix_power(self.koopman_operator_, step) @ X_encoded

    def predict_time(self, time):
        # modes @ exp(diag(log(eigenvalues)/sample_rate) * time) @ initial_condition
        return self.modes_ @ np.diag(np.exp(np.log(self.eigenvalues_)/self.sample_rate * time)) \
            @ self.initial_condition_
