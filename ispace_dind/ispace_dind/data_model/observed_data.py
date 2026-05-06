from dataclasses import dataclass
import numpy as np

@dataclass
class ObservedData:
    coord: np.ndarray
    ray: np.ndarray
    center_coord: np.ndarray

@dataclass
class ObservedPersonData(ObservedData):
    coord: np.ndarray
    ray: np.ndarray
    center_coord: np.ndarray
    visual_conf: float
    bbox: np.ndarray
    bbox_conf: float
    keypoints: np.ndarray
    keypoints_conf: np.ndarray
    data: str=''