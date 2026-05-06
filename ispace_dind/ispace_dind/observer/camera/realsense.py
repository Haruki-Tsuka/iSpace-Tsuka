from .camera_base import CameraBase
import numpy as np
import os

from ispace_dind.utils.realsense_manager import RealSenseManager
from ispace_dind.utils.coords_converter import CoordsConverter

class RealsenseCamera(CameraBase):

    def __init__(self):
        super().__init__()
        self.rs_manager = RealSenseManager()
        if os.path.exists('camera_matrix'):
            rvec = np.load('camera_matrix/rvecs.npy')
            tvec = np.load('camera_matrix/tvecs.npy')
            mtx = np.load('camera_matrix/mtx.npy')
            dist = np.load('camera_matrix/dist.npy')
        else:
            os.makedirs('camera_matrix')
            raise FileNotFoundError('カメラ行列が存在しません。')
        self.coords_converter = CoordsConverter(tvec, rvec, mtx, dist)

    def get_img(self) -> np.ndarray:
        return self.rs_manager.get_img()

    def get_depth(self) -> np.ndarray:
        return self.rs_manager.get_depth_numpy()
    
    def get_depth_and_img(self):
        return self.get_depth(), self.get_img()
    
    def update(self):
        self.rs_manager.update()
    
    def get_coords_converter(self) -> CoordsConverter:
        return self.coords_converter
    
    def get_3d_coordinate(self, pixel_x, pixel_y, depth=None):
        return self.rs_manager.get_3d_coordinate(pixel_x, pixel_y, depth)