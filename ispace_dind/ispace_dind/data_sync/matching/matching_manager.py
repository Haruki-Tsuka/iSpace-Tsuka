from typing import List
from ispace_dind.data_model.observed_data import ObservedData
from ispace_dind.data_sync.matching.matching import global_matching
from typing import Dict, Tuple
from ispace_dind.data_model.sync_data import SyncData
import numpy as np
from ispace_dind.data_sync.matching.matching import face_matching

class MatchingManager:

    
    def __init__(self, node):
        self.node = node
        self.current_matching_dict:Dict[int, List[Tuple[str, SyncData, float, np.ndarray]]] = {} # 観測データのインデックス, ユニークID, 受信データ, 確信度, 中点
        self.matching_list:List[Dict[int, Tuple[str, SyncData, float, np.ndarray]]] = []
        self.dsu_dict:Dict[int, str] = {}
        self.last_id_confs = {}
        
    def matching(self, observed_data_list:List[ObservedData], frame):
        self.current_matching_dict = {}
        self.matching_list = []
        self.last_id_confs = {}
        dind_data_list = self.node.dind_data_dict.values()
        for dind_data in dind_data_list:
            received_data_list:List[SyncData] = dind_data.get_latest_data()
            if received_data_list is None:
                continue
            for received_data in received_data_list:
                self.last_id_confs[f'{dind_data.hostname}_{received_data.local_id}'] = received_data.visual_conf
                dsu_id = received_data.mapping_json.get(self.node.hostname, None)
                if dsu_id is not None:
                    dsu_id = int(dsu_id)
                    self.dsu_dict[dsu_id] = f"{dind_data.hostname}_{received_data.local_id}"
                    
            matching_results = global_matching(self.node, dind_data, received_data_list, observed_data_list, frame)
            if matching_results is None:
                continue
            matching_dict = {}
            for observed_idx, recieved_data, softmax_conf, mid_point in matching_results:
                unique_id = f"{dind_data.hostname}_{recieved_data.local_id}"
                current_matching_list = self.current_matching_dict.get(observed_idx, [])
                current_matching_list.append((unique_id, recieved_data, softmax_conf, mid_point))
                self.current_matching_dict[observed_idx] = current_matching_list
                matching_dict[observed_idx] = (unique_id, recieved_data, softmax_conf, mid_point)
            self.matching_list.append(matching_dict)
            
    def face_matching(self, observed_data_list:List[ObservedData], frame):
        self.current_matching_dict = {}
        self.matching_list = []
        dind_data_list = self.node.dind_data_dict.values()
        for dind_data in dind_data_list:
            received_data_list:List[SyncData] = dind_data.get_latest_data()
            if received_data_list is None:
                continue
            matching_results = face_matching(self.node, dind_data, received_data_list, observed_data_list, frame)
            if matching_results is None:
                continue
            for observed_idx, recieved_data, softmax_conf, mid_point in matching_results:
                observed_data = observed_data_list[observed_idx]
                if observed_data.data == '':
                    continue
                send_data = f"{recieved_data.hostname}_{recieved_data.local_id}_{observed_data.data}"
                self.node.ros_interface.publish_face_data(send_data)
                
            
    def get_matching_data(self, observed_idx:int) -> List[Tuple[str, SyncData, float, np.ndarray]]:
        return self.current_matching_dict.get(observed_idx, [])
    
    def get_matching_list(self) -> List[List[Tuple[int, str, SyncData, float, np.ndarray]]]:
        return self.matching_list
    
    def get_dsu_dict(self) -> Dict[int, str]:
        return self.dsu_dict
    
    def get_visual_conf(self, unique_id:str) -> float:
        return self.last_id_confs.get(unique_id, 0.0)