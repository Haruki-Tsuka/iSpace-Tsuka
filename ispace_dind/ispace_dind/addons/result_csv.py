from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
from ispace_dind.utils.file_manager import CSVFileManager
import datetime

@addon
class ResultCsv(AddonBase):

    def __init__(self, node):
        self.node = node
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.csv_file_manager = CSVFileManager(dir='results_csv', csv_name=f'results_{self.start_time}.csv', columns=['timestamp', 'id', 'x1', 'y1', 'x2', 'y2'])
        self.csv_file_manager.create()
        self.last_timestamp = 0

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.write_csv)
        
    def write_csv(self, data):
        timestamp = data['timestamp']
        if timestamp == self.last_timestamp:
            print('timestamp is the same')
            return
        self.last_timestamp = timestamp
        for tracker in data['update_trackers']:
            if tracker.observed_data.bbox is None:
                continue
            if tracker.get_local_id() <= 0 or tracker.sync_state > 0:
                continue
            self.csv_file_manager.add([timestamp, tracker.get_local_id(), tracker.observed_data.bbox[0], tracker.observed_data.bbox[1], tracker.observed_data.bbox[2], tracker.observed_data.bbox[3]])