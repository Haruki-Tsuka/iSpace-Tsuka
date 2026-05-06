from .id_manager import IDManager
from .tracker import Tracker
from typing import List, Dict, Tuple
from ispace_dind.data_model.observed_data import ObservedData, ObservedPersonData
from .tracker import Tracker
import copy
import numpy as np

class TrackerManager:
    
    def __init__(self):
        self.id_manager = IDManager()
        self.proposed_tracker_list:List[Tracker] = []
        self.active_tracker_dict:Dict[int, Tracker] = {}
        self.lost_tracker_dict:Dict[int, Tracker] = {}
        
    def update_ekf(self, tracker: Tracker, observed_data: ObservedData):
        tracker.update(observed_data)
        if tracker.get_local_id() < 0:
            tracker.set_local_id(self.id_manager.get_next_id())
        
    def get_assosiate_tracker_list(self):
        tracker_list = list(self.active_tracker_dict.values()) + copy.deepcopy(self.proposed_tracker_list)
        self.proposed_tracker_list.clear()
        return tracker_list
    
    def predict_all_trackers(self):
        tracker_list = list(self.active_tracker_dict.values())
        for tracker in tracker_list:
            if tracker.predict():
                continue
            self.set_lost_tracker(tracker.get_local_id())
        return tracker_list
            
    def propose_new_tracker(self, observed_data_idx: int, observed_data_list: List[ObservedPersonData]):
        observed_data = observed_data_list[observed_data_idx]
        if observed_data.bbox_conf < 0.7:
            return None
        for o_idx, o_data in enumerate(observed_data_list):
            if o_idx == observed_data_idx:
                continue
            if np.linalg.norm(observed_data.center_coord - o_data.center_coord) < 30:
                return None
        new_tracker = Tracker(observed_data)
        new_tracker.set_local_id(self.id_manager.get_next_pre_id())
        self.proposed_tracker_list.append(new_tracker)
        return new_tracker
    
    def update_tracker(self, tracker: Tracker):
        if tracker.get_local_id() < 0:
            return
        if tracker.get_local_id() in self.active_tracker_dict:
            self.active_tracker_dict[tracker.get_local_id()] = tracker
        elif tracker.get_local_id() in self.lost_tracker_dict:
            del self.lost_tracker_dict[tracker.get_local_id()]
            self.active_tracker_dict[tracker.get_local_id()] = tracker
        else:
            self.active_tracker_dict[tracker.get_local_id()] = tracker

    '''
    local_idを引数としてTrackerを取得する
    返り値:
        is_active: bool
        tracker: Tracker
    is_activeがTrueの場合、trackerは現在アクティブなトラッカーである。
    is_activeがFalseの場合、trackerは過去にアクティブだったトラッカーである。
    trackerがNoneの場合、トラッカーが存在しない。
    '''
    def get_tracker(self, local_id: int) -> Tuple[bool, Tracker]:
        if local_id in self.active_tracker_dict:
            return True, self.active_tracker_dict[local_id]
        elif local_id in self.lost_tracker_dict:
            return False, None
        else:
            return False, None
        
    def is_active_tracker(self, local_id: int) -> bool:
        if local_id in self.active_tracker_dict:
            return True
        else:
            return False
    
    def is_lost_tracker(self, local_id: int) -> bool:
        if local_id in self.lost_tracker_dict:
            return True
        else:
            return False
        
    def set_lost_tracker(self, local_id: int):
        if local_id < 0:
            return
        if local_id in self.active_tracker_dict:
            self.lost_tracker_dict[local_id] = self.active_tracker_dict[local_id]
            del self.active_tracker_dict[local_id]
            
    def switch_tracker(self, id_map_dict: Dict[int, int]):
        switched_list = []
        for old_id, new_id in id_map_dict.items():
            if old_id == new_id:
                continue
            if old_id < 0 or old_id in switched_list:
                continue
            switched_list.append(new_id)
            old_tracker = self.get_tracker(old_id)[1]
            new_tracker = self.get_tracker(new_id)[1]
            old_tracker.set_local_id(new_id)
            self.update_tracker(old_tracker)
            if new_tracker is not None:
                new_tracker.set_local_id(old_id)
                self.update_tracker(new_tracker)
            else:
                self.remove_tracker(old_id)
            
    def remove_tracker(self, local_id: int):
        if local_id in self.active_tracker_dict:
            del self.active_tracker_dict[local_id]
        elif local_id in self.lost_tracker_dict:
            del self.lost_tracker_dict[local_id]