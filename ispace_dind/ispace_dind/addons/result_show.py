from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import cv2
import matplotlib.pyplot as plt

@addon
class ResultShow(AddonBase):

    def __init__(self, node):
        self.node = node
        cmap = plt.get_cmap('tab20')
        self.colors = [[int(c * 255) for c in cmap(i)[:3]] for i in range(cmap.N)]

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.publish_marker)

    def publish_marker(self, data):
        img = self.node.camera_img
        dsu = data['dsu']
        for tracker in data['update_trackers']:
            if tracker.observed_data.center_coord is None or tracker.sync_state > 0:
                continue
            color = (255, 255, 255)
            if self.node.hostname == 'mn3':
                tracker_id = tracker.get_local_id()
                color = self.colors[tracker_id % len(self.colors)]
            else:
                unique_ids = dsu.get_unique_ids_from_local(tracker.get_local_id())
                for unique_id in unique_ids:
                    hostname = unique_id.split('_')[0]
                    if hostname == 'mn3':
                        color = self.colors[int(unique_id.split('_')[1]) % len(self.colors)]
                        break
            
            cv2.rectangle(img, tuple(map(int, tracker.observed_data.bbox[0:2])), tuple(map(int, tracker.observed_data.bbox[2:4])), color, 2)
            cv2.putText(img, f'{tracker.local_id}', tuple(map(int, tracker.observed_data.center_coord)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
            self.node.camera_img = img