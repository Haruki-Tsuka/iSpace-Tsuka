from ispace_dind.data_sync.tracking.tracker import Tracker
from rclpy.node import Node
from ispace_interfaces.msg import DindData, PointDataArray
from ispace_dind.ros_bridge.message_utils import f32multi2numpy, numpy2f32multi, ros_now_msg
from ispace_dind.utils.coords_converter import CoordsConverter
from ispace_dind.data_model.dind_data import DINDData
from ispace_dind.data_model.sync_data import SyncData
from typing import List
import numpy as np
import json
from std_msgs.msg import String

class ROSInterface:
    
    def __init__(self, node: Node):
        self.node = node
        self.dind_data_pub = self.node.create_publisher(DindData, 'dind_data', 10)
        self.point_data_array_pub = self.node.create_publisher(PointDataArray, 'point_data_array', 10)
        self.point_data_array_sub = self.node.create_subscription(PointDataArray, 'point_data_array', self.point_data_array_callback, 10)
        self.dind_data_sub = self.node.create_subscription(DindData, 'dind_data', self.dind_data_callback, 10)
        self.dind_data_timer = self.node.create_timer(1.0, self.publish_dind_data)
        self.face_data_pub = self.node.create_publisher(String, 'face_data', 10)
        
    def publish_dind_data(self):
        dind_data = DindData()
        dind_data.hostname = self.node.hostname
        dind_data.frame_width = self.node.frame_width
        dind_data.frame_height = self.node.frame_height
        dind_data.tvec = numpy2f32multi(self.node.coords_converter.tvec)
        dind_data.rvec = numpy2f32multi(self.node.coords_converter.rvec)
        dind_data.camera_matrix = numpy2f32multi(self.node.coords_converter.camera_matrix)
        dind_data.dist_coeffs = numpy2f32multi(self.node.coords_converter.dist_coeffs)
        self.dind_data_pub.publish(dind_data)
        
    
    def dind_data_callback(self, msg: DindData):
        if msg.hostname not in self.node.dind_data_dict:
            self.node.dind_data_dict[msg.hostname] = DINDData(hostname=msg.hostname,
                                                         coords_converter=CoordsConverter(f32multi2numpy(msg.tvec), f32multi2numpy(msg.rvec), f32multi2numpy(msg.camera_matrix), f32multi2numpy(msg.dist_coeffs)),
                                                         frame_width=msg.frame_width,
                                                         frame_height=msg.frame_height
                                                         )
            
    def publish_sync_data(self, tracker_list: List[Tracker], dsu):
        point_data_array = PointDataArray()
        for tracker in tracker_list:
            point_data = tracker.to_msg()
            dsu_ids = dsu.get_unique_ids_from_local(tracker.get_local_id())
            dsu_mapping_json = {}
            for dsu_id in dsu_ids:
                dsu_mapping_json[dsu_id.split('_')[0]] = dsu_id.split('_')[1]
            point_data.mapping_json = json.dumps(dsu_mapping_json)
            point_data_array.point_array.append(point_data)
        point_data_array.hostname = self.node.hostname
        point_data_array.stamp = ros_now_msg()
        self.point_data_array_pub.publish(point_data_array)
        
    def point_data_array_callback(self, msg: PointDataArray):
        if msg.hostname == self.node.hostname:
            return
        if msg.hostname not in self.node.dind_data_dict:
            return
        sync_data_list: List[SyncData] = []
        timestamp = msg.stamp
        point_data_list = msg.point_array
        for point_data in point_data_list:
            sync_data = SyncData(local_id=point_data.track_id,
                                 coord=np.array([point_data.coord.x, point_data.coord.y, point_data.coord.z]),
                                 ray=np.array([point_data.ray.x, point_data.ray.y, point_data.ray.z]),
                                 state=point_data.state,
                                 visual_conf=point_data.visual_conf,
                                 assosiate_conf=point_data.assosiate_conf,
                                 data=point_data.data,
                                 mapping_json=json.loads(point_data.mapping_json))
            sync_data_list.append(sync_data)
        self.node.dind_data_dict[msg.hostname].add_data(timestamp, sync_data_list)
        
    def publish_face_data(self, data: str):
        face_data = String()
        face_data.data = data
        self.face_data_pub.publish(face_data)