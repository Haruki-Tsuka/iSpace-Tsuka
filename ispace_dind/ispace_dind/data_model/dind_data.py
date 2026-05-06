from dataclasses import dataclass, field
from ispace_dind.utils.coords_converter import CoordsConverter
from ispace_dind.data_model.sync_data import SyncData
from ispace_dind.ros_bridge.message_utils import time2int
from typing import List
from rclpy.time import Time

@dataclass
class DINDData:   
    hostname: str
    coords_converter: CoordsConverter
    frame_width: int
    frame_height: int
    
    recieved_data: dict=field(default_factory=dict)
    
    last_timestamp: Time=None
    before_timestamp: Time=None
    
    def add_data(self, timestamp: Time,
                     data_list: List[SyncData],
                     max_length: int=100) -> None:
        timestamp = time2int(timestamp)
        self.recieved_data[timestamp] = data_list
        self.last_timestamp = timestamp
        if len(self.recieved_data) > max_length:
            del self.recieved_data[next(iter(self.recieved_data))]        
            
    """
    最新のデータを取得する。
    最新のデータが取得できない場合はNoneを返す。（データが空の状態と区別）
    
    Returns:
        List[SyncData]: 最新のデータ
        None: 最新のデータが取得できない場合
    """
    def get_latest_data(self) -> List[SyncData]:
        if self.last_timestamp != self.before_timestamp:
            self.before_timestamp = self.last_timestamp
            return self.recieved_data.get(self.last_timestamp)
        else:
            return None