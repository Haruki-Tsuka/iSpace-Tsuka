from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import cv2
import matplotlib.pyplot as plt
from ispace_dind.utils.file_manager import CSVFileManager
from ispace_dind.observer.dataset_observer import DsObserver
import rclpy
import numpy as np
import datetime

@addon
class SetPerm(AddonBase):

    def __init__(self, node):
        self.node = node
        cmap = plt.get_cmap('tab20')
        self.colors = [[int(c * 255) for c in cmap(i)[:3]] for i in range(cmap.N)]
        #HHMMSS
        self.start_time = datetime.datetime.now().strftime('%H%M%S')
        self.csv_file_manager = CSVFileManager(dir='csv', csv_name=f'perm_data_{self.start_time}.csv', columns=['timestamp', 'id', 'match_id', 'perm', 'x', 'y', 'z'])
        self.csv_file_manager.create()

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.set_perm)

    def set_perm(self, data):
        timestamp = 0
        if self.node.observer is not None and isinstance(self.node.observer, DsObserver):
            timestamp = self.node.observer.last_timestamp
        img = data['frame']
        #imgの下に40pixel分黒い背景を伸ばす
        for kalman in data['update_kalmans']:
            self.csv_file_manager.add([timestamp, kalman.tracker_id, str(kalman.point_data.get_dind_set_ids()), kalman.inherited_dind_id, kalman.get_coord()[0], kalman.get_coord()[1], kalman.get_coord()[2]])
            print('ADD', timestamp, kalman.tracker_id, str(kalman.point_data.get_dind_set_ids()), kalman.inherited_dind_id, kalman.get_coord()[0], kalman.get_coord()[1], kalman.get_coord()[2])
            if kalman.inherited_dind_id == 'PERM':
                cv2.rectangle(img, tuple(map(int, kalman.get_bbox()[0:2])), tuple(map(int, kalman.get_bbox()[2:4])), (0, 0, 255), 4)