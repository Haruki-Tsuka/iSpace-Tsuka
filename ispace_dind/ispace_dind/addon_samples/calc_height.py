from ispace_dind.addons.addon_base import AddonBase
from ispace_dind.utils.event_handler import Event
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import numpy as np
import cv2

class CalcHeight(AddonBase):

    def __init__(self, node):
        self.node = node

    def register(self):
        self.node.event_handler.add_listener(Event.YOLO_EVENT, self.calc_height)
        self.marker_pub = self.node.create_publisher(Marker, 'marker_sync', 10)

    def calc_height(self, results, has_data):
        img = results.plot()
        cam_xy = np.array([8.89871152, 0.19959212])
        cam_z = 2.76286758
        lines = []
        cam_P = Point(x=cam_xy[0], y=cam_xy[1], z=cam_z)
        if has_data:
            for xy, xyxy in zip(results.keypoints.xy.numpy(), results.boxes.xyxy.numpy()):
                foot = (xy[15] + xy[16]) / 2
                if foot[0] < 0.01 and foot[1] < 0.01:
                    pass
                foot = np.array([foot[0], xyxy[3]])
                mask = xy[:5,0] != 0
                if np.any(mask):
                    ex = int(np.mean(xy[:5,0][mask]))
                    ey = int(np.mean(xy[:5,1][mask]))
                else:
                    ex, ey = 0, 0
                if ex < 0.01 and ey < 0.01:
                    continue
                head = np.array([ex, ey])
                head = np.array([head[0], xyxy[1]])
                world_coords = self.node.coords_converter.pixel2world(np.array([foot, head, np.array([ex,ey])]), world_z=0)
                x1, y1 = cam_xy
                x2, y2 = world_coords[1,0:2]
                xp, yp = world_coords[0,0:2]
                dx = x2 - x1
                dy = y2 - y1
                t = ((xp - x1) * dx + (yp - y1) * dy) / (dx**2 + dy**2)
                Qx = x1 + t * dx
                Qy = y1 + t * dy
                head2Q = np.sqrt((x2 - Qx)**2 + (y2 - Qy)**2)
                cam2head = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                height = cam_z*(head2Q/cam2head)
                Q_shadow = self.node.coords_converter.world2pixel(np.array([[Qx, Qy, 0]]))
                head_shadow = self.node.coords_converter.world2pixel(np.array([[x2, y2, 0]]))
                cam_shadow = self.node.coords_converter.world2pixel(np.array([[cam_xy[0], cam_xy[1], 0]]))
                head_P = Point(x=world_coords[2,0], y=world_coords[2,1], z=0.0)
                lines.extend([cam_P, head_P])
                cv2.line(img, (int(head_shadow[0,0]), int(head_shadow[0,1])), (int(cam_shadow[0,0]), int(cam_shadow[0,1])), (0, 0, 255), 1)
                cv2.circle(img, (int(Q_shadow[0,0]), int(Q_shadow[0,1])), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(head_shadow[0,0]), int(head_shadow[0,1])), 5, (0, 0, 255), -1)
                cv2.putText(img, f'{height:.2f}', (int(head[0]), int(head[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            # Marker の作成（ここでは LINE_LIST を使用）
            marker = Marker()
            marker.header.frame_id = "map"  # 必要に応じてTFフレーム名を変更
            marker.ns = "isp"
            marker.id = 1001
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            # 線分の太さ（scale.x で指定）
            marker.scale.x = 0.05

            marker.points = lines

            # Marker の色を設定（例：黄色）
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self.marker_pub.publish(marker)
        cv2.imshow('result', img)
        cv2.waitKey(1)

def register_addon(node):
    addon = CalcHeight(node)
    addon.register()
    return addon