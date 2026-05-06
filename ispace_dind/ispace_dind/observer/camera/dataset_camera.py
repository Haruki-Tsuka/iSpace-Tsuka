from .camera_base import CameraBase
import numpy as np
from ispace_dind.utils.coords_converter import CoordsConverter
from ispace_dind.ros_bridge.message_utils import ros_now_msg
import os
from std_msgs.msg import Int64
import cv2

class DatasetCamera(CameraBase):

    def __init__(self, node, dataset_dir: str, start_time=None):
        super().__init__()
        timestamp = self.__get_timestamp_number()
        
        self.img_dir = os.path.join(dataset_dir, f'img')
        self.depth_dir = os.path.join(dataset_dir, f'depth')
        self.matrix_dir = os.path.join(dataset_dir, f'camera_matrix')
        
        #dirが存在しない場合エラー
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f'imgディレクトリが存在しません。: {self.img_dir}')
        if not os.path.exists(self.depth_dir):
            raise FileNotFoundError(f'depthディレクトリが存在しません。: {self.depth_dir}')
        if not os.path.exists(self.matrix_dir):
            raise FileNotFoundError(f'camera_matrixディレクトリが存在しません。: {self.matrix_dir}')
        
        file_list = os.listdir(self.img_dir)
        self.image_numbers = [int(file_name.split('.')[0]) for file_name in file_list if file_name.endswith('.jpg')]
        self.image_numbers.sort()
        
        self.dataset_timestamp = self.image_numbers[0]
        
        rvec = np.load(os.path.join(self.matrix_dir, f'rvecs.npy'))
        tvec = np.load(os.path.join(self.matrix_dir, f'tvecs.npy'))
        mtx = np.load(os.path.join(self.matrix_dir, f'mtx.npy'))
        dist = np.load(os.path.join(self.matrix_dir, f'dist.npy'))
        self.coords_converter = CoordsConverter(tvec, rvec, mtx, dist)
        
        self.start_time = start_time
        self.diff = None
        if start_time is None:
            self.start_time_sub = node.create_subscription(Int64, 'start_time', self.start_time_callback, 10)
        elif start_time < 0:
            self.start_time = timestamp - start_time
            self.start_time_pub = node.create_publisher(Int64, 'start_time', 10)
            self.diff = self.start_time - self.image_numbers[0]
            self.last_timestamp = timestamp
            node.create_timer(1, self.publish_start_time)
            
    def __get_timestamp_number(self):
        now = ros_now_msg()
        timestamp = int(f'{now.sec}{str(now.nanosec//1000000).zfill(3)}')
        return timestamp
            
    def start_time_callback(self, msg: Int64):
        self.start_time = msg.data
        if self.diff is None:
            self.diff = self.start_time - self.image_numbers[0]
            
    def publish_start_time(self):
        msg = Int64()
        msg.data = self.start_time
        self.start_time_pub.publish(msg)

    def get_img(self) -> np.ndarray:
        return cv2.imread(os.path.join(self.img_dir, f'{self.dataset_timestamp}.jpg'))

    def get_depth(self) -> np.ndarray:
        return np.load(os.path.join(self.depth_dir, f'{self.dataset_timestamp}.npy'))
    
    def update(self):
        if self.diff is None:
            return
        self.last_timestamp = self.__get_timestamp_number() - self.diff - 2000
        self.dataset_timestamp = min(self.image_numbers, key=lambda x: abs(x - self.last_timestamp))
        
    
    def get_coords_converter(self) -> CoordsConverter:
        return self.coords_converter
    
    def get_3d_coordinate(self, pixel_x, pixel_y, depth=None):
        pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        # P=None だと「正規化座標」(x_n, y_n) が返る (z=1 平面)
        und = cv2.undistortPoints(pts, self.coords_converter.camera_matrix, self.coords_converter.dist_coeffs, P=None)  # shape=(1,1,2)
        x_n, y_n = und[0, 0]
        Z = float(depth)
        return np.array([x_n * Z, y_n * Z, Z], dtype=np.float32)
    
    def get_depth_and_img(self):
        depth = np.load(os.path.join(self.depth_dir, f'{self.dataset_timestamp}.npy'))
        img = cv2.imread(os.path.join(self.img_dir, f'{self.dataset_timestamp}.jpg'))
        return img, depth
    
    def get_timestamp(self):
        return self.dataset_timestamp