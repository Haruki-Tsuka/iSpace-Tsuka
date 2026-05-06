from typing import List, Dict
from ispace_dind.data_sync.tracking.tracker import Tracker
from ispace_dind.data_model.observed_data import ObservedData
import numpy as np

class OnlineChineseRestaurantProcess:
    
    def __init__(self):
        self.prob_dict:Dict[int, float] = {}
    
    def calculate_prob(self, tracker_list:List[Tracker], observed_data_list:List[ObservedData], maharanobis_matrix:np.ndarray):
        self.prob_dict.clear()
        cov_matrix = np.eye(3)*0.5
        total_assosiate_times = sum([tracker.assosiate_times for tracker in tracker_list]) + len(observed_data_list)
        alpha = 0.5
        denominator = alpha + total_assosiate_times - 1
        for observed_data_idx in range(len(observed_data_list)):
            P_max = -1.0
            P_new = 1/np.sqrt((2*np.pi)**3 * np.linalg.det(cov_matrix))*(alpha/denominator)
            for tracker_idx in range(len(tracker_list)):
                tracker = tracker_list[tracker_idx]
                P_k = (1/np.sqrt((2*np.pi)**3 * np.linalg.det(tracker.ekf.get_P()[0:3,0:3]))) * np.exp(-0.5 * maharanobis_matrix[tracker_idx, observed_data_idx]**2)
                P_k *= (tracker.assosiate_times/denominator)
                if P_max < P_k:
                    P_max = P_k
                    
            if P_new > P_max:
                self.prob_dict[observed_data_idx] = P_new
            else:
                self.prob_dict[observed_data_idx] = 0.0
                
    def get_prob(self, observed_data_idx:int) -> float:
        return 1.0
        return self.prob_dict.get(observed_data_idx, 0.0)