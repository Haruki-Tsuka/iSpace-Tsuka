from ispace_dind.data_sync.base import DataSync
import numpy as np
from ispace_dind.utils.event_handler import Event
import time
import cv2
from ispace_dind.data_model.observed_data import ObservedPersonData
from typing import List

from ispace_dind.data_sync.matching.matching_manager import MatchingManager
from ispace_dind.data_sync.tracking.tracker_manager import TrackerManager
from ispace_dind.data_sync.assignment.assignment import get_maharanobis_matrix, lap_from_cost
from ispace_dind.data_sync.assignment.cost_matrix import softmax_cost_matrix
from ispace_dind.data_sync.matching.dsu import DSU
from ispace_dind.data_sync.tracking.tracker import TrackerState, Tracker
from ispace_dind.ros_bridge.message_utils import ros_now_sec

class ImprovedRaySync(DataSync):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.matching_manager = MatchingManager(node)
        self.person_tracker_manager = TrackerManager()
        self.dsu = DSU()
        
        self.timestamp = time.time()
        
    def is_in_frame(self, coord:np.ndarray, frame_height:int, frame_width:int) -> bool:
        pixel_coord = self.node.coords_converter.world2pixel(coord)
        # np.array([[x, y], [x, y]])のように座標が格納されているので、xとyの値をそれぞれチェックし、True/Falseのarrayを返す
        x = pixel_coord[0,0]
        y = pixel_coord[0,1]

        # 画素の包含判定（0 <= x < W, 0 <= y < H）
        return (x >= 0) & (x < frame_width) & (y >= 0) & (y < frame_height)

    def _maybe_add_new_tracker(
        self,
        observed_data_idx:int,
        observed_data_list:List[ObservedPersonData],
        update_trackers:List[Tracker],
        assigned_observed_data_idxs_list:List[int],
    ) -> None:
        new_kalman_data = self.person_tracker_manager.propose_new_tracker(observed_data_idx, observed_data_list)
        if new_kalman_data is not None:
            update_trackers.append(new_kalman_data)
            assigned_observed_data_idxs_list.append(observed_data_idx)
        
    def update_assign_trackers(self, tracker_list:List[Tracker], observed_data_list:List[ObservedPersonData], assigned_tracker_idxs:List[int], assigned_observed_data_idxs:List[int]):
        update_trackers = []
        for assigned_tracker_idx, assigned_observed_data_idx in zip(assigned_tracker_idxs, assigned_observed_data_idxs):
            tracker = tracker_list[assigned_tracker_idx]
            observed_data = observed_data_list[assigned_observed_data_idx]
            center = observed_data.center_coord
            if center[0] < 20 or center[0] > self.node.frame_width - 20 or center[1] < 20 or center[1] > self.node.frame_height - 20:
                tracker.state = TrackerState.TENTATIVE
            self.person_tracker_manager.update_ekf(tracker, observed_data)
            update_trackers.append(tracker)
        return update_trackers
    
    def assignment(self, cost_matrix:np.ndarray, tracker_indices:List[int], threshold_max:float):
        assigned_tracker_idxs, assigned_observed_data_idxs, cost_matrix = lap_from_cost(cost_matrix[tracker_indices, :], threshold_max=threshold_max)
        return tracker_indices[assigned_tracker_idxs], assigned_observed_data_idxs, cost_matrix

    def clicked(self, event, x, y, flags, param):
        self.node.event_handler.emit(Event.CLICKED_EVENT, event, x, y, flags, param)
        
    def assosiate_data(self, observed_data_list:List[ObservedPersonData]):
        self.person_tracker_manager.predict_all_trackers()
        update_trackers = []
        if len(observed_data_list) == 0:
            frame = self.node.camera.get_img()
            observer_timestamp = self.node.observer.get_last_timestamp()
            event_data = {
                'update_trackers': update_trackers,
                'frame': frame,
                'dsu': self.dsu,
                'timestamp': observer_timestamp
            }
            self.node.event_handler.emit(Event.DATA_SYNC_EVENT, event_data)
            return
        
        frame = self.node.camera.get_img()
        tracker_list = self.person_tracker_manager.get_assosiate_tracker_list()

        # ------------------------------------------------------------------
        # 1) Matching
        # ------------------------------------------------------------------
                
        self.matching_manager.matching(observed_data_list, frame)
        dsu_dict = self.matching_manager.get_dsu_dict()
        for dsu_id, dsu_name in dsu_dict.items():
            self.dsu.add_unique_id(dsu_name, dsu_id)
        
        # ------------------------------------------------------------------
        # 2) Cost Matrix
        # ------------------------------------------------------------------
        maharanobis_matrix = get_maharanobis_matrix(tracker_list, observed_data_list)
        tracker_conf_matrix = softmax_cost_matrix(maharanobis_matrix)
        
        # ------------------------------------------------------------------
        # 3) Two-stage Assignment
        # ------------------------------------------------------------------
        assigned_result_dict = {}
        assigned_tracker_idxs_list = []
        assigned_observed_data_idxs_list = []
        high_priority_tracker_indices = [idx for idx, tracker in enumerate(tracker_list) if len(tracker_conf_matrix) > 0 and tracker.ekf.get_predicted_seconds() < 0.5 and tracker_conf_matrix[idx] > 0.5 and tracker.assosiate_times > 2]
        
        assigned_tracker_idxs = []
        assigned_observed_data_idxs = []
        if len(high_priority_tracker_indices) > 0:
            assigned_tracker_idxs, assigned_observed_data_idxs, _ = lap_from_cost(maharanobis_matrix[high_priority_tracker_indices, :], threshold_max=0.8)
            assigned_tracker_idxs = [high_priority_tracker_indices[idx] for idx in assigned_tracker_idxs]
            update_trackers.extend(self.update_assign_trackers(tracker_list, observed_data_list, assigned_tracker_idxs, assigned_observed_data_idxs))
            assigned_result_dict = dict(zip([tracker_list[idx].get_local_id() for idx in assigned_tracker_idxs], assigned_observed_data_idxs))
            assigned_observed_data_idxs_list.extend(assigned_observed_data_idxs)
            assigned_tracker_idxs_list.extend(assigned_tracker_idxs)
            
        low_priority_tracker_indices = [idx for idx in range(len(tracker_list)) if idx not in assigned_tracker_idxs]
        low_priority_observed_data_indices = [idx for idx in range(len(observed_data_list)) if idx not in assigned_observed_data_idxs]
        
        if len(low_priority_tracker_indices) > 0 and len(low_priority_observed_data_indices) > 0:
            assigned_tracker_idxs, assigned_observed_data_idxs, _ = lap_from_cost(maharanobis_matrix[low_priority_tracker_indices][:, low_priority_observed_data_indices], threshold_max=2.0)
            assigned_tracker_idxs = [low_priority_tracker_indices[idx] for idx in assigned_tracker_idxs]
            assigned_observed_data_idxs = [low_priority_observed_data_indices[idx] for idx in assigned_observed_data_idxs]
            update_trackers.extend(self.update_assign_trackers(tracker_list, observed_data_list, assigned_tracker_idxs, assigned_observed_data_idxs))
            #tracker_list[assigned_tracker_idx].get_local_id()をKeyとして、observed_data_idxの対応関係を辞書に格納（1対1対応保証）
            assigned_result_dict.update(dict(zip([tracker_list[idx].get_local_id() for idx in assigned_tracker_idxs], assigned_observed_data_idxs)))
            assigned_observed_data_idxs_list.extend(assigned_observed_data_idxs)
            assigned_tracker_idxs_list.extend(assigned_tracker_idxs)
            
        no_assignment_tracker_idxs = [idx for idx in range(len(tracker_list)) if idx not in assigned_tracker_idxs_list]
            
        #tracker_listからassigned_tracker_idxs_listに含まれないindexのtrackerを削除
        for no_assignment_tracker in [tracker for index, tracker in enumerate(tracker_list) if index not in assigned_tracker_idxs_list]:
            if not self.is_in_frame(no_assignment_tracker.ekf.get_x()[0:3], frame.shape[0], frame.shape[1]):
                self.person_tracker_manager.remove_tracker(no_assignment_tracker.get_local_id())

        # ------------------------------------------------------------------
        # 4) Vote-based Matching / New Tracker Proposal
        # ------------------------------------------------------------------
        matching_results = {}
        matching_data_dict = {}
        can_spawn_new_tracker = len(tracker_list) < len(observed_data_list)
        #全てのDINDからのデータがどのローカルIDに割り当てられているかを投票で決定
        for observed_data_idx, observed_data in enumerate(observed_data_list):
            matching_data_list = self.matching_manager.get_matching_data(observed_data_idx)
            if len(matching_data_list) == 0 and can_spawn_new_tracker and observed_data_idx not in assigned_observed_data_idxs_list: #マッチング結果がない場合、新規ID生成確率が高い場合は、新規ID生成として扱う
                self._maybe_add_new_tracker(observed_data_idx, observed_data_list, update_trackers, assigned_observed_data_idxs_list)
                continue
            vote_dict = {} # {local_id: (total_conf, [matching_data,...], [mid_point,...]), ...}
            for matching_unique_id, matching_data, matching_conf, mid_point in matching_data_list:
                #マッチング結果が違う場合、mid_pointを用いて距離ベースの比較によって最終的な割当先を決定
                # if matching_conf < 0.8:
                #     continue
                matching_local_id = self.dsu.get_local_id_from_unique(matching_unique_id)
                if matching_local_id is None:
                    matching_local_id = -1
                
                if matching_local_id not in vote_dict:
                    vote_dict[matching_local_id] = (matching_conf, [matching_data], [mid_point], [matching_unique_id])
                else:
                    vote_dict[matching_local_id][0] += matching_conf
                    vote_dict[matching_local_id][1].append(matching_data)
                    vote_dict[matching_local_id][2].append(mid_point)
                    vote_dict[matching_local_id][3].append(matching_unique_id)
            
            if len(vote_dict) > 0:
                max_id = max(vote_dict.keys())
                if can_spawn_new_tracker and observed_data_idx not in assigned_observed_data_idxs_list:
                    self._maybe_add_new_tracker(observed_data_idx, observed_data_list, update_trackers, assigned_observed_data_idxs_list)
                    continue
                matching_results[observed_data_idx] = max_id
                matching_data_dict[observed_data_idx] = vote_dict[max_id]
            else:
                if can_spawn_new_tracker and observed_data_idx not in assigned_observed_data_idxs_list:
                    self._maybe_add_new_tracker(observed_data_idx, observed_data_list, update_trackers, assigned_observed_data_idxs_list)
        
        id_map_dict = {} # {old_id: new_id, ...}
            
        # ------------------------------------------------------------------
        # 5) Update Tracker States / Switching
        # ------------------------------------------------------------------
        #Tracker基準で割り当てしていく方が真っ当だよね
        for assigned_observed_data_idx, assigned_tracker_idx in zip(assigned_observed_data_idxs_list, assigned_tracker_idxs_list):
            tracker = tracker_list[assigned_tracker_idx]
            assign_conf = tracker_conf_matrix[assigned_tracker_idx]
            max_id = matching_results.get(assigned_observed_data_idx, None)
            if max_id is None:
                continue
            tracker.update_state(assign_conf, max_id)
            if tracker.get_state() == TrackerState.TENTATIVE:
                if assign_conf > tracker.STATE_THRESHOLD:
                    if max_id < 0:
                        tracker.state = TrackerState.CONFIRMED
                        for matching_unique_id in matching_data_dict[assigned_observed_data_idx][3]:
                            if '-' not in matching_unique_id:
                                self.dsu.add_unique_id(matching_unique_id, tracker.get_local_id())
                    else:
                        tracker.state = TrackerState.CONFIRMED
                        id_map_dict[tracker.get_local_id()] = max_id
            elif tracker.get_state() == TrackerState.CONFIRMED:
                if assign_conf > tracker.STATE_THRESHOLD and max_id != tracker.get_local_id():
                   before_assignment = assigned_result_dict.get(max_id, None)
                   if before_assignment is not None:
                       id_map_dict[tracker.get_local_id()] = max_id
                   else:
                       _, compare_tracker = self.person_tracker_manager.get_tracker(max_id)
                       if compare_tracker is not None and compare_tracker.get_state() != TrackerState.LOCKED:
                           id_map_dict[tracker.get_local_id()] = max_id
                           
        for tracker in update_trackers:
            self.person_tracker_manager.update_tracker(tracker)
            
        self.person_tracker_manager.switch_tracker(id_map_dict)
        
        # ------------------------------------------------------------------
        # 6) Temporary Permission Control
        # ------------------------------------------------------------------
        # for tracker in update_trackers:
        #     uids = self.dsu.get_unique_ids_from_local(tracker.get_local_id())
        #     if len(uids) == 0:
        #         tracker.perm = True
        #         continue
        #     for uid in uids:
        #         visual_conf = self.matching_manager.get_visual_conf(uid)
        #         if tracker.observed_data.visual_conf < visual_conf:
        #             tracker.perm = False
        #         else:
        #             tracker.perm = True
        
        # ------------------------------------------------------------------
        # 7) Build Output Trackers
        # ------------------------------------------------------------------
        for no_assignment_tracker_idx in no_assignment_tracker_idxs:
            tracker = tracker_list[no_assignment_tracker_idx]
            tracker.sync_state = 1
            tracker.observed_data.ray = tracker.ekf.get_x()[0:3]
            update_trackers.append(tracker)
            
        no_assignment_observed_data_idxs = [idx for idx in range(len(observed_data_list)) if idx not in assigned_observed_data_idxs_list]
        for no_assignment_observed_data_idx in no_assignment_observed_data_idxs:
            temp_tracker = Tracker(observed_data_list[no_assignment_observed_data_idx])
            temp_tracker.sync_state = 2
            temp_tracker.track_id = -999
            update_trackers.append(temp_tracker)
        
        # ------------------------------------------------------------------
        # 8) Publish / Event / Visualize
        # ------------------------------------------------------------------
        self.node.ros_interface.publish_sync_data(update_trackers, self.dsu)
        observer_timestamp = self.node.observer.get_last_timestamp()
        event_data = {'update_trackers':update_trackers, 'frame':frame, 'dsu':self.dsu, 'timestamp' : observer_timestamp}
        self.node.event_handler.emit(Event.DATA_SYNC_EVENT, event_data)
        #print(f'ASSOSIATE TIME: {time.time() - self.timestamp:.3f} sec ({1/(time.time() - self.timestamp):.3f}fps)')
        self.timestamp = time.time()
