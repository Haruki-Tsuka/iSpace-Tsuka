from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import kachaka_api
import threading
import cv2
import numpy as np

@addon
class CallKachaka(AddonBase):

    COUNT_THRESHOLD = 30
    TRACK_CHECK_COUNT = 20

    def __init__(self, node):
        self.node = node
        self.client = kachaka_api.KachakaApiClient('192.168.1.183:26400')
        self.hand_up_count = {}
        self.tracked_id = -1
        self.count = 0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.end_count = 0

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.call_kachaka)

    def is_hand_up(self, kalman):
        keypoints = kalman.get_keypoints()
        if keypoints is None:
            return False 
        if keypoints[8,1] < 0.1 or keypoints[10,1] < 0.1:
            return False
        return (kalman.get_nose()[1] > (keypoints[8,1]+keypoints[10,1])/2)
    
    def kachaka_call(self, x, y, msg=None):
        if msg is not None:
            self.client.speak(msg)
        self.client.move_to_pose(x, y, 0.0)

    def speak_kachaka(self, msg):
        self.client.speak(msg)

    def call_kachaka(self, data):
        frame = data['frame']
        self.end_count += 0
        self.count += 1
        for kalman in data['update_kalmans']:
            if kalman.point_data.inherited_data == 'TRACKING' and self.tracked_id == -1:
                self.tracked_id = kalman.tracker_id
            if kalman.get_nose() is None:
                continue
            if kalman.tracker_id <= -1:
                continue
            print('TRACKING_ID: ', self.tracked_id)
            if kalman.tracker_id == self.tracked_id:
                cv2.rectangle(frame, tuple(map(int, kalman.get_bbox()[0:2])), tuple(map(int, kalman.get_bbox()[2:4])), (0, 255, 0), 2)
                # cv2.putText(frame, f'{kalman.tracker_id}:{kalman.inherited_data}:{kalman.inherited_dind_id}', tuple(kalman.get_nose()), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, tuple(map(int, kalman.get_bbox()[0:2])), tuple(map(int, kalman.get_bbox()[2:4])), (255, 255, 255), 2)
                # cv2.putText(frame, f'{kalman.tracker_id}', tuple(kalman.get_nose()), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if self.tracked_id == -1 and kalman.inherited_dind_id == 'PERM':
                self.end_count = 0
                self.count = 0
                #挙手を一定時間した人を検知してkachakaを呼ぶ
                count_hand_up = self.hand_up_count.get(kalman.tracker_id, 0)
                if self.is_hand_up(kalman):
                    count_hand_up += 1
                else:
                    count_hand_up = 0
                self.hand_up_count[kalman.tracker_id] = count_hand_up
                #BBOXに透過したCountの進捗バーを重ねる
                cv2.rectangle(frame, (int(kalman.get_bbox()[0]), int(kalman.get_bbox()[1]+(kalman.get_bbox()[3]-kalman.get_bbox()[1])*(1-count_hand_up/self.COUNT_THRESHOLD))), tuple(map(int, kalman.get_bbox()[2:4])), (255, 255, 255, 0.5), -1)
                if count_hand_up >= self.COUNT_THRESHOLD:
                    self.tracked_id = kalman.tracker_id
                    self.goal_x = kalman.ekf.x[0]
                    self.goal_y = kalman.ekf.x[1]
                    kalman.inherited_data = 'TRACKING'
                    threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y, f'ID{kalman.tracker_id}の追跡を開始します。')).start()
                    break
                    
            else:
                if kalman.tracker_id == self.tracked_id and kalman.inherited_dind_id == 'PERM':
                    self.end_count = 0
                    if self.count > self.TRACK_CHECK_COUNT:
                        self.count = 0
                        if np.sqrt((self.goal_x-kalman.ekf.x[0])**2+(self.goal_y-kalman.ekf.x[1])**2) > 1.5:
                            self.goal_x = kalman.ekf.x[0]
                            self.goal_y = kalman.ekf.x[1]
                            threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y)).start()
                            print('===================')
                            print('KACHAKA_CALL', self.goal_x, self.goal_y)
                            print('===================')
                            break
                if self.end_count >= 30:
                    self.tracked_id = -1
                    self.end_count = 0
                    threading.Thread(target=self.speak_kachaka, args=('ターゲットを見失いました。')).start()

                    

