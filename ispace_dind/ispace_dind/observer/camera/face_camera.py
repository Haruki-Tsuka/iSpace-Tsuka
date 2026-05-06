import cv2
from .camera_base import CameraBase
import numpy as np

class FaceCamera(CameraBase):
    
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

    def get_img(self) -> np.ndarray:
        ret, frame = self.cap.read()
        return frame
    
    def get_depth(self) -> np.ndarray:
        return None
    
    def update(self):
        pass