from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event
import kachaka_api
import threading
import cv2
import numpy as np
from geometry_msgs.msg import Point

@addon
class BringCandy(AddonBase):

    COUNT_THRESHOLD = 30
    TRACK_CHECK_COUNT = 100
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
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.bring_candy)
        
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

    def is_hand_up(self, kalman):
        keypoints = kalman.get_keypoints()
        if keypoints is None:
            return False 
        if keypoints[8,1] < 0.1 or keypoints[10,1] < 0.1:
            return False
        return (kalman.get_nose()[1] > (keypoints[8,1]+keypoints[10,1])/2)
    
    def kachaka_call(self, x, y, msg=None):
        if self.cool_time < 0:
            return
        self.publish_goal_point(x, y)
        if msg is not None:
            self.client.speak(msg)
        self.client.move_to_pose(x, y, 0.0)

    def speak_kachaka(self, msg, omake=0):
        if self.cool_time > 0:
            print('===================')
            print('SPEAK_KACHAKA', msg)
            print('===================')
            self.cool_time = -self.COOL_TIME
            self.publish_goal_point(self.goal_x, self.goal_y, -100.0)
            self.client.speak(msg, cancel_all=True)

    def bring_candy(self, data):
        frame = data['frame']
        self.end_count += 0
        self.count += 1
        self.cool_time += 1
        if self.tracked_id != -1:
            goal_pixel = self.node.coords_converter.world2pixel(np.array([[self.goal_x, self.goal_y, 0.12]]))
            cv2.circle(frame, tuple(map(int, goal_pixel[0])), 10, (0, 0, 255), 2)
        if self.tracked_id != -1 and self.detect_count % 5 == 0:
            self.kachaka_pose = self.client.get_robot_pose()
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
                    threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y, f'ハッピーハロウィン！お菓子を届けます！')).start()
                    break
                    
            else:
                self.detect_count += 1
                if kalman.tracker_id == self.tracked_id and kalman.inherited_dind_id == 'PERM':
                    self.end_count = 0
                    if self.count > self.TRACK_CHECK_COUNT:
                        self.count = 0
                        if np.sqrt((self.goal_x-kalman.ekf.x[0])**2+(self.goal_y-kalman.ekf.x[1])**2) > 2.0:
                            self.goal_x = kalman.ekf.x[0]
                            self.goal_y = kalman.ekf.x[1]
                            threading.Thread(target=self.kachaka_call, args=(self.goal_x, self.goal_y)).start()
                            print('===================')
                            print('KACHAKA_CALL', self.goal_x, self.goal_y)
                            print('===================')
                            break
                if self.detect_count % 5 and kalman.inherited_dind_id == 'PERM':
                    if self.kachaka_pose is None:
                        continue
                    if np.sqrt((self.kachaka_pose.x-kalman.ekf.x[0])**2+(self.kachaka_pose.y-kalman.ekf.x[1])**2) < 0.8:
                        xm = self.kachaka_pose.x - self.DETECT_AREA/2
                        ym = self.kachaka_pose.y - self.DETECT_AREA/2
                        xp = self.kachaka_pose.x + self.DETECT_AREA/2
                        yp = self.kachaka_pose.y + self.DETECT_AREA/2
                        zm = self.CANDY_HIGHT
                        zp = self.CANDY_HIGHT + self.DETECT_AREA/2
                        candy_pixel = self.node.coords_converter.world2pixel(np.array([[xm, ym, zm], [xp, ym, zm], [xm, yp, zm], [xm, ym, zp], [xp, yp, zm], [xp, ym, zp], [xm, yp, zp], [xp, yp, zp]]))
                        hands = kalman.get_keypoints()[[9,10]]
                        cv2.rectangle(frame, tuple(map(int, candy_pixel[0])), tuple(map(int, candy_pixel[1])), (0, 255, 0), 2)
                        #candy_pixelのbbox範囲内に、handsのどちらかが存在するかを判定
                        good_message = "美味しく召し上がってください。"
                        bad_message = "キャンディーを勝手に取らないでください。"
                        if np.min(candy_pixel[:,0]) < hands[0,0] < np.max(candy_pixel[:,0]) and np.min(candy_pixel[:,1]) < hands[0,1] < np.max(candy_pixel[:,1]):
                            if self.tracked_id == kalman.tracker_id:
                                threading.Thread(target=self.speak_kachaka, args=(good_message, 1)).start()
                                break
                            else:
                                threading.Thread(target=self.speak_kachaka, args=(bad_message, 1)).start()
                                break
                        elif np.min(candy_pixel[:,0]) < hands[1,0] < np.max(candy_pixel[:,0]) and np.min(candy_pixel[:,1]) < hands[1,1] < np.max(candy_pixel[:,1]):
                            if self.tracked_id == kalman.tracker_id:
                                threading.Thread(target=self.speak_kachaka, args=(good_message, 1)).start()
                                break
                            else:
                                threading.Thread(target=self.speak_kachaka, args=(bad_message, 1)).start()
                                break
                if self.end_count >= 30:
                    self.tracked_id = -1
                    self.end_count = 0
                    threading.Thread(target=self.speak_kachaka, args=('ターゲットを見失いました。', 1)).start()
                    

                    

