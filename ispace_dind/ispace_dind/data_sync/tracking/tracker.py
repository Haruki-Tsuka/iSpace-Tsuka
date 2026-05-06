from ispace_dind.data_sync.tracking.new_ekf import EKF3D
from ispace_dind.data_model.observed_data import ObservedData
from ispace_interfaces.msg import PointData
from geometry_msgs.msg import Point
import json
import numpy as np

class TrackerState:
    TENTATIVE = 0  # 仮割り当て。信頼度が低い（ID修正を積極的に許可）
    CONFIRMED = 1  # 確定割り当て。信頼度が高い（ID修正を慎重に許可）
    LOCKED = 2  # ロック割り当て。信頼度が高い（ID修正を許可しない）

class Tracker:
    
    STATE_THRESHOLD = 0.8
    COUNT_MAX = 7
    COUNT_LOCKED_THRESHOLD = 3  # 確定割り当てが連続で一定数以上続いた場合、ロック状態にする
    
    def __init__(self, observed_data: ObservedData):
        self.local_id :int = -1
        self.observed_data :ObservedData = observed_data
        self.assosiate_times :int = 1
        self.predicted_times :int = 0
        self.assosiate_conf :float = 0.0
        self.ekf :EKF3D = EKF3D(observed_data.coord)
        self.state :TrackerState = TrackerState.TENTATIVE
        self.confirmed_count :int = 0
        self.data :str = ''
        self.sync_state :int = 0
        self.perm = True
        
    def predict(self):
        self.predicted_times += 1
        self.ekf.predict()
        self.sync_state = 1
        if self.predicted_times > 1 and self.state == TrackerState.LOCKED:
            self.state = TrackerState.CONFIRMED
        return (self.ekf.get_predicted_seconds() < 2.0)
        
    def update(self, observed_data: ObservedData):
        self.observed_data = observed_data
        self.assosiate_times += 1
        self.predicted_times = 0
        self.ekf.update(observed_data.coord)
        self.sync_state = 0
        
    def get_local_id(self):
        return self.local_id
    
    def set_local_id(self, local_id: int):
        self.local_id = local_id
        self.state = TrackerState.TENTATIVE
        
    def get_state(self):
        return self.state
    
    def update_state(self, assign_conf: float, matched_id: int):
        #assign_confにnp.array([1.0])が入ることがある。その場合にfloatに変換したい
        if isinstance(assign_conf, np.ndarray):
            assign_conf = assign_conf[0]
        self.assosiate_conf = assign_conf
        if self.state == TrackerState.LOCKED:
            if assign_conf > self.STATE_THRESHOLD:
                self.confirmed_count -= 1 if self.confirmed_count > 0 else 0
            else:
                self.confirmed_count += 1
                
            if self.confirmed_count > 2:
                self.state = TrackerState.CONFIRMED
                self.confirmed_count = 0
        elif self.state == TrackerState.CONFIRMED:
            if assign_conf < self.STATE_THRESHOLD:
                self.confirmed_count -= 1 if self.confirmed_count > 0 else 0
            else:
                if self.local_id == matched_id:
                    self.confirmed_count += 1
                    
                if self.confirmed_count > self.COUNT_LOCKED_THRESHOLD:
                    self.state = TrackerState.LOCKED
                    self.confirmed_count = 0
                    
    def to_msg(self) -> PointData:
        point_data = PointData()
        point_data.track_id = self.local_id
        point_data.coord = Point(x=self.ekf.get_x()[0], y=self.ekf.get_x()[1], z=self.ekf.get_x()[2])
        point_data.ray = Point(x=self.observed_data.ray[0], y=self.observed_data.ray[1], z=self.observed_data.ray[2])
        point_data.visual_conf = self.observed_data.visual_conf
        point_data.assosiate_conf = self.assosiate_conf
        point_data.data = self.data
        return point_data
    
    