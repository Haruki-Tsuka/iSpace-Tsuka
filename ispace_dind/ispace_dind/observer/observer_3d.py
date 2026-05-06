from ispace_dind.utils.event_handler import Event
from ispace_dind.observer.base import Observer
from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional
from ispace_dind.data_model.observed_data import ObservedPersonData
from ispace_dind.observer.camera.camera_base import CameraBase

class Observer3D(Observer):
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
        self.yolo = YOLO(node.config_dict.get('config.yolo_model', 'yolo11m-pose.engine'))
        

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
        img, depth = self.node.camera.get_img_and_depth()
        results = self.yolo(img, verbose=False, conf=0.1)
        has_data, bbox, bbox_conf, kp, kp_conf = self.get_yolo_data(results)
        
        self.node.event_handler.emit(Event.YOLO_EVENT, results[0], has_data)
        if not has_data:
            return None
        
        # 鼻座標をベースとしたデータのフィルタリング処理
        nose_coords = self.get_nose_code(kp)
        nose_coords = self.nose_filter(nose_coords, kp_conf, threshold=11)
        coords_3d, select_idxs = self.get_3d_coords(nose_coords, depth)
        bbox = bbox[select_idxs]
        nose_coords = nose_coords[select_idxs]
        kp_conf = kp_conf[select_idxs]
        # 重みの計算
        iou_array = self.get_iou_array(bbox)
        kp_weight = self.get_kp_weight(kp_conf)
        if len(coords_3d) > 0:
            weights = iou_array * kp_weight * ((10-coords_3d[:,2])/10.0)
        else:
            weights = iou_array * kp_weight
        
        world_coords = self.node.camera.get_coords_converter().camera2world(coords_3d)
        ray_coords = self.node.camera.get_coords_converter().pixel2world(nose_coords, 0.0)
        observed_data_list = []
        for coord, ray_coord, weight, bbox, bbox_conf, kp, kp_conf, nose_coord in zip(world_coords, ray_coords, weights, bbox, bbox_conf, kp, kp_conf, nose_coords):
            observed_data_list.append(ObservedPersonData(coord=coord,
                                                 ray=ray_coord,
                                                 center_coord=nose_coord,
                                                 visual_conf=weight,
                                                 bbox_conf=bbox_conf,
                                                 keypoints=kp,
                                                 keypoints_conf=kp_conf,
                                                 bbox=bbox))
        
        return observed_data_list
    
    def get_3d_coords(self, pixel_coords: np.ndarray, depth: np.ndarray, area: int = 5, limit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        2Dピクセル座標と深度画像から3D座標を計算します。
        指定されたピクセル周辺の深度値の中央値を使用して3D座標を計算します。

        Args:
            pixel_coords: 2Dピクセル座標配列 [N, 2]
            depth: 深度画像 [H, W]
            area: 深度計算のための周辺領域サイズ（デフォルト: 5）
            limit: 画像端からの最小距離（デフォルト: 10）

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D座標配列 [M, 3]。Mは有効な座標の数（N以下）
        """
        coords_3d = []
        select_idxs = []
        for i in range(pixel_coords.shape[0]):
            x, y = pixel_coords[i]
            if x < limit or x >= depth.shape[1]-limit:
                continue
            #ｘ−エリアからｘ＋エリア、ｙ−エリアからｙ＋エリアまでの深度の中央値を求める。ただし、範囲が領域外に出ないように
            x_min = max(area, x - area)
            x_max = min(depth.shape[1]-area, x + area)
            y_min = max(area, y - area)
            y_max = min(depth.shape[0]-area, y + area)
            depth_area = depth[y_min:y_max, x_min:x_max]
            nonzero_depth = depth_area[depth_area != 0]
            # Calculate the median of the non-zero values
            median_depth = np.median(nonzero_depth)
            coord_3d = self.node.camera.get_3d_coordinate(x, y, median_depth)
            if coord_3d is None:
                continue
            else:
                select_idxs.append(i)
                coords_3d.append(coord_3d)
        return np.array(coords_3d, dtype=np.float32), select_idxs
    
    def get_last_timestamp(self):
        return self.node.camera.get_timestamp()