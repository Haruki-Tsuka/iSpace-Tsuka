from dataclasses import dataclass
import numpy as np

@dataclass
class PointData:
    coord: np.ndarray
    ray: np.ndarray
    visual_conf: float
    assosiate_conf: float
    keypoints: np.ndarray
    nose: np.ndarray
    bbox: np.ndarray
    data: str