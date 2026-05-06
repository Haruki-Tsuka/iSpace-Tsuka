import numpy as np
from typing import Tuple
from ispace_dind.ros_bridge.message_utils import ros_now_sec

class CameraBase:

    def __init__(self):
        pass

    def get_img(self) -> np.ndarray:
        pass

    def get_depth(self) -> np.ndarray:
        pass

    def get_img_and_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_img(), self.get_depth()
    
    def get_timestamp(self) -> float:
        return ros_now_sec()
    
    def update(self):
        pass
    
    def set_img(self, img: np.ndarray):
        pass