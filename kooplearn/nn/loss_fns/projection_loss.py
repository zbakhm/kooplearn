import torch


def projection_loss(x_encoded, y_encoded):
    cov_x = x_encoded @ x_encoded.T
    cov_y = y_encoded @ y_encoded.T
    cov_xy = x_encoded @ y_encoded.T

