from ispace_dind.utils.event_handler import Event
from ispace_dind.observer.base import Observer
from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional
from ispace_dind.data_model.observed_data import ObservedPersonData
from ispace_dind.observer.camera.camera_base import CameraBase

class ObserverFace(Observer):
    """
    RealSenseカメラを使用した観測クラス。
    RealSenseカメラからのRGB画像と深度画像を使用して人物の検出と追跡を行います。
    """

    def __init__(self, node):
        """
        Args:
            node: ROS2ノードインスタンス
        """
        super().__init__(node)
        self.yolo = YOLO("yolo11m-pose.engine")
        self.camera = node.camera
        

    def observe(self) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        観測を実行し、結果を返します。
        以下の処理を行います：
        1. RGB画像と深度画像の取得
        2. YOLOによる人物検出
        3. 鼻の座標の取得とフィルタリング
        4. 3D座標の計算
        5. 重みの計算
        6. クラスタリングと可視化

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[dict]]:
                - 元画像
                - 処理済み画像（YOLOの検出結果を可視化）
                - 観測データ（座標、重み、距離など）
        """
        self.node.camera.update()
        img = self.node.camera.get_img()
        results = self.yolo(img, verbose=False, conf=0.1)
        has_data, bbox, bbox_conf, kp, kp_conf = self.get_yolo_data(results)
        self.node.event_handler.emit(Event.YOLO_EVENT, results[0], has_data)
        if not has_data:
            print('No data')
            return img, img, None
        nose_coords = self.get_nose_code(kp)
        nose_coords = self.nose_filter(nose_coords, kp_conf, threshold=11)
        ray_coords = self.node.camera.get_coords_converter().pixel2world(nose_coords, 2.0)
        iou_array = self.get_iou_array(bbox)
        kp_weight = self.get_kp_weight(kp_conf)
        weights = iou_array * kp_weight
            
        self.node.event_handler.emit(Event.IMAGE_GET_EVENT, img)
        observed_data_list = []
        for ray_coord, nose_coord, bbox, weight, bbox_conf, kp, kp_conf in zip(ray_coords, nose_coords, bbox, weights, bbox_conf, kp, kp_conf):
            observed_data_list.append(ObservedPersonData(coord=None,
                                                 ray=ray_coord,
                                                 center_coord=nose_coord,
                                                 visual_conf=weight,
                                                 bbox_conf=bbox_conf,
                                                 keypoints=kp,
                                                 keypoints_conf=kp_conf,
                                                 bbox=bbox))
        
        data = {'frame':img,
                'observed_data_list':observed_data_list}
        return img, img, data