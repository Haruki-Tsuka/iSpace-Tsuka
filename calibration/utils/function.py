import numpy as np
from .extend_kalman_filter import EKF_2D

def euclidean_distance(point_a:np.ndarray, point_b:np.ndarray) -> np.float64:
    return np.linalg.norm(point_a - point_b)

def euclidean_distance_array(point_as:np.ndarray, point_bs:np.ndarray) ->np.ndarray:
    return np.linalg.norm(point_as - point_bs, axis=1)

def mahalanobis_distance(tracker:EKF_2D, coord) -> np.float64:
    coord = np.array([coord.x, coord.y])
    return tracker.mahalanobis_distance(coord)