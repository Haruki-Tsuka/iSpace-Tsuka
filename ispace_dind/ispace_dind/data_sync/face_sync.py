from ispace_dind.data_sync.base import DataSync
import numpy as np
from ispace_dind.utils.event_handler import Event
import time
import cv2
from ispace_dind.data_model.observed_data import ObservedPersonData
from typing import List

from ispace_dind.data_sync.matching.matching_manager import MatchingManager
from ispace_dind.ros_bridge.message_utils import ros_now_sec

class FaceSync(DataSync):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.matching_manager = MatchingManager(node)
        self.timestamp = time.time()
    
    def assosiate_data(self, data):
        if data is None:
            return
        
        frame = data.get('frame')
        observed_data_list:List[ObservedPersonData] = data.get('observed_data_list')
        
        if observed_data_list is None:
            #TODO 空のデータを送信する
            self.node.event_handler.emit(Event.DATA_SYNC_EVENT, None, observed_data_list)
            return
                
        # --------------------------------------------------------
        
        self.matching_manager.face_matching(observed_data_list, frame)
        event_data = {'frame':frame}
        self.node.event_handler.emit(Event.DATA_SYNC_EVENT, event_data)
        
        frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
        cv2.imshow('result', frame)
        cv2.waitKey(1)    
