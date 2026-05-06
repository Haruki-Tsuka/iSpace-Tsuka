from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import cv2
import matplotlib.pyplot as plt
from std_msgs.msg import String

@addon
class ShowName(AddonBase):

    def __init__(self, node):
        self.node = node
        cmap = plt.get_cmap('tab20')
        self.colors = [[int(c * 255) for c in cmap(i)[:3]] for i in range(cmap.N)]
        self.face_data_dict = {}

    def register(self):
        self.node.create_subscription(String, 'face_data', self.subscribe_face_data, 10)
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.publish_marker)
        
    def subscribe_face_data(self, data):
        id = data.data
        dind = id.split('_')[0]
        if dind != self.node.hostname:
            return
        local_id = id.split('_')[1]
        name = id.split('_')[2]
        self.face_data_dict[local_id] = name

    def publish_marker(self, data):
        img = data['frame']
        for tracker in data['update_trackers']:
            if tracker.observed_data.center_coord is None or tracker.sync_state > 0:
                continue
            
            if tracker.local_id not in self.face_data_dict:
                cv2.putText(img, f'{self.face_data_dict[tracker.local_id]}', tuple(map(int, tracker.observed_data.center_coord)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.rectangle(img, tuple(map(int, tracker.observed_data.bbox[0:2])), tuple(map(int, tracker.observed_data.bbox[2:4])), (255, 255, 255), 2)
            