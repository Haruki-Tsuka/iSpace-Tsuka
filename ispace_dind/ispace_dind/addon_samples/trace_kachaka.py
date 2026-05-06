from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import kachaka_api
import threading
import cv2
import numpy as np
from geometry_msgs.msg import Point

@addon
class TraceKachaka(AddonBase):

    COUNT_THRESHOLD = 30
    TRACK_CHECK_COUNT = 50
    CANDY_HIGHT = 0.45
    DETECT_AREA = 0.4
    COOL_TIME = 60

    def __init__(self, node):
        self.node = node
        self.client = kachaka_api.KachakaApiClient('192.168.1.164:26400')
        self.hand_up_count = {}
        self.tracked_id = -1
        self.count = 0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.end_count = 0
        self.detect_count = 0
        self.kachaka_pose = None
        self.cool_time = 0

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.trace_kachaka)
        
        self.goal_point_pub = self.node.create_publisher(Point, 'goal_point', 10)
        self.goal_point_sub = self.node.create_subscription(Point, 'goal_point', self.goal_point_callback, 10)

    def goal_point_callback(self, msg):
        if msg.z < -1.0:
            self.count = 0
            self.cool_time = -self.COOL_TIME
            return
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.count = 0
        
    def publish_goal_point(self, x, y, z=0.0):
        msg = Point()
        msg.x = x
        msg.y = y
        msg.z = z
        self.goal_point_pub.publish(msg)

    def is_hand_up(self, tracker):
        keypoints = tracker.observed_data.keypoints
        if keypoints is None:
            return False 
        if keypoints[8,1] < 0.1 or keypoints[10,1] < 0.1:
            return False  
        return (tracker.observed_data.center_coord[1] > (keypoints[8,1]+keypoints[10,1])/2)
    
    def kachaka_call(self, x, y):
        if self.cool_time < 0:
            return
        self.publish_goal_point(x, y)
        self.client.move_to_pose(x, y, 0.0)

    def speak_kachaka(self, msg, omake=0):
        if self.cool_time > 0:
            print('===================')
            print('SPEAK_KACHAKA', msg)
            print('===================')
            self.cool_time = -self.COOL_TIME
            self.publish_goal_point(self.goal_x, self.goal_y, -100.0)
            self.client.speak(msg, cancel_all=True)

    def trace_kachaka(self, data):
        frame = data['frame']
        self.end_count += 0
        self.count += 1
        self.cool_time += 1
        if self.tracked_id != -1 and self.detect_count % 5 == 0:
            self.kachaka_pose = self.client.get_robot_pose()
        for tracker in data['update_trackers']:
            if tracker.data == 'TRACKING' and self.tracked_id == -1:
                self.tracked_id = tracker.get_local_id()
            if tracker.get_local_id() == self.tracked_id:
                cv2.rectangle(frame, tuple(map(int, tracker.observed_data.bbox[0:2])), tuple(map(int, tracker.observed_data.bbox[2:4])), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, tuple(map(int, tracker.observed_data.bbox[0:2])), tuple(map(int, tracker.observed_data.bbox[2:4])), (255, 255, 255), 2)
            if self.tracked_id == -1 and tracker.perm:
                self.end_count = 0
                self.count = 0
                #挙手を一定時間した人を検知してkachakaを呼ぶ
                count_hand_up = self.hand_up_count.get(tracker.get_local_id(), 0)
                if self.is_hand_up(tracker):
                    count_hand_up += 1
                else:
                    count_hand_up = 0
                self.hand_up_count[tracker.get_local_id()] = count_hand_up
                #BBOXに透過したCountの進捗バfーを重ねる
                cv2.rectangle(frame, (int(tracker.observed_data.bbox[0]), int(tracker.observed_data.bbox[1]+(tracker.observed_data.bbox[3]-tracker.observed_data.bbox[1])*(1-count_hand_up/self.COUNT_THRESHOLD))), tuple(map(int, tracker.observed_data.bbox[2:4])), (255, 255, 255, 0.5), -1)
                if count_hand_up >= self.COUNT_THRESHOLD:
                    self.tracked_id = tracker.get_local_id()
                    self.goal_x = tracker.ekf.ekf.x[0]
                    self.goal_y = tracker.ekf.ekf.x[1]
                    tracker.data = 'TRACKING'
                    threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y)).start()
                    break
                    
            else:
                self.detect_count += 1
                if tracker.get_local_id() == self.tracked_id and tracker.perm:
                    self.end_count = 0
                    if self.count > self.TRACK_CHECK_COUNT:
                        self.count = 0
                        if np.sqrt((self.goal_x-tracker.ekf.ekf.x[0])**2+(self.goal_y-tracker.ekf.ekf.x[1])**2) > 2.0:
                            self.goal_x = tracker.ekf.ekf.x[0]
                            self.goal_y = tracker.ekf.ekf.x[1]
                            threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y)).start()
                            break
                if self.end_count >= 30:
                    self.tracked_id = -1
                    self.end_count = 0
                    threading.Thread(target=self.speak_kachaka, args=('ターゲットを見失いました。', 1)).start()
                    

                    

