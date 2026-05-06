from ispace_dind.utils.realsense_manager import RealSenseManager
from ispace_dind.utils.event_handler import EventHandler, Event
from ispace_dind.observer.base import Observer
from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional
from ispace_dind.data_model.observed_data import ObservedPersonData
from ispace_dind.observer.camera.camera_base import CameraBase
import datetime
from ispace_dind.utils.file_manager import CSVFileManager
import cv2

class ObserverYolo(Observer):
    """
    RealSenseカメラを使用した観測クラス。
    RealSenseカメラからのRGB画像と深度画像を使用して人物の検出と追跡を行います。
    """

    def __init__(self, node, camera: CameraBase):
        """
        Args:
            node: ROS2ノードインスタンス
        """
        super().__init__(node)
        self.yolo = YOLO("yolo11m-pose.engine")
        self.camera = camera
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M')
        self.csv_file_manager = CSVFileManager(dir='results_csv', csv_name=f'results_{self.start_time}.csv', columns=['timestamp', 'id', 'x1', 'y1', 'x2', 'y2'])
        self.csv_file_manager.create()
        self.last_timestamp = 0
        

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
        self.camera.update()
        img = self.camera.get_img()
        # results = self.yolo(img, verbose=False, conf=0.1)
        results = self.yolo.track(
            source=img,
            persist=True,
            conf=0.1,
            iou=0.5,
            verbose=False,
            tracker='botsort.yaml'
        )
        
        timestamp = self.camera.get_timestamp()
        if timestamp == self.last_timestamp:
            print('timestamp is the same')
            return img, img, None
        self.last_timestamp = timestamp
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return img, img, None
        boxes = result.boxes
        if boxes.id is not None:
            for box in zip(boxes.id, boxes.xyxy):
                track_id = int(box[0].item())
                x1, y1, x2, y2 = box[1].tolist()
                self.csv_file_manager.add([self.camera.get_timestamp(), track_id, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                #cv2.putText(img, f'{track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
        
        #cv2.imshow('img', img)
        #cv2.waitKey(1)
        return img, img, None
    
    def get_last_timestamp(self):
        return self.camera.get_timestamp()
