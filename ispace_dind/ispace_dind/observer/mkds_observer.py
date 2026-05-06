from ispace_dind.utils.realsense_manager import RealSenseManager
from ispace_dind.utils.event_handler import EventHandler, Event
from ispace_dind.observer.base import Observer
from ultralytics import YOLO
import numpy as np
import cv2
import os
import datetime
import rclpy

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class MKDSObserver(Observer):

    def __init__(self, node):
        super().__init__(node)
        self.rs_manager = RealSenseManager()
        mk_dir('dataset')
        self.data_dir_path = 'dataset/' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
        mk_dir(self.data_dir_path)
        mk_dir(self.data_dir_path + '/img')
        mk_dir(self.data_dir_path + '/depth')
        self.last_timestamp = 0

    def get_depth_and_img(self):
        self.rs_manager.update()
        return self.rs_manager.get_img(), self.rs_manager.get_depth_numpy()
    
    def observe(self):
        img, depth = self.get_depth_and_img()
        now = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.ROS_TIME).now().to_msg()
        time_path = f'{now.sec}{str(now.nanosec//1000000).zfill(3)}'
        img_path = f'{self.data_dir_path}/img/{time_path}.jpg'
        depth_path = f'{self.data_dir_path}/depth/{time_path}.npy'
        cv2.imwrite(img_path, img)
        np.save(depth_path, depth)
        cv2.imshow('img', img)
        cv2.waitKey(10)
        return img, img, None