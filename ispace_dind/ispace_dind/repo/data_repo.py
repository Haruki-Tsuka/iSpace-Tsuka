from typing import Any
from ispace_dind.ros_bridge.message_utils import ros_now_sec


class DataRepo:
    
    def __init__(self):
        self.timestamp = ros_now_sec()
        self.data_dict = {}
    
    def update_timestamp(self):
        self.timestamp = ros_now_sec()
    
    def add_data(self, local_id: int, data: Any):
        self.data_dict[local_id] = data
        
    def get_data(self, local_id: int):
        return self.data_dict.get(local_id, None)
    
    def get_all_data(self):
        return self.data_dict