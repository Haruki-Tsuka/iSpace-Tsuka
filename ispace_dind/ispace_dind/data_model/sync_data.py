from dataclasses import dataclass
import numpy as np
from typing import List
from typing import Tuple
from ispace_interfaces.msg import PointData
from geometry_msgs.msg import Point
import json
from dataclasses import field

@dataclass
class SyncData:
    local_id: int
    
    coord: np.ndarray
    ray: np.ndarray
    visual_conf: float=0.0
    assosiate_conf: float=-1.0
    state: int=0
    
    data: str=''
    mapping_json: dict=field(default_factory=dict)
    
    def to_msg(self) -> PointData:
        point_data = PointData()
        point_data.track_id = self.local_id
        point_data.state = self.state
        point_data.coord = Point(x=self.coord[0], y=self.coord[1], z=self.coord[2])
        point_data.ray = Point(x=self.ray[0], y=self.ray[1], z=self.ray[2])
        point_data.visual_conf = self.visual_conf
        point_data.assosiate_conf = self.assosiate_conf
        point_data.data = self.data
        point_data.mapping_json = json.dumps(self.mapping_json)
        return point_data