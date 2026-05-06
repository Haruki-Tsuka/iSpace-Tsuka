from ispace_dind.addons.addon_base import AddonBase, addon
import numpy as np
import cv2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import math

@addon
class CameraMarker(AddonBase):

    def __init__(self, node):
        self.node = node

    def register(self):
        self.marker = None
        self.marker_pub = self.node.create_publisher(Marker, 'marker_sync', 10)
        self.node.create_multi_threaded_timer(1.0, self.publish_camera_fov_marker)

    def create_camera_fov_marker(self, tvec: np.ndarray, 
                             rvec: np.ndarray, 
                             depth: float, 
                             fov: tuple) -> Marker:
        # fov の分解（水平・垂直画角）
        fov_horizontal, fov_vertical = fov

        # カメラ座標系（原点：カメラ位置、x軸：正面、y軸：右方向、z軸：上方向）での画像平面の四隅を計算
        half_width  = depth * math.tan(fov_horizontal / 2)
        half_height = depth * math.tan(fov_vertical   / 2)

        # カメラ座標系での各点（原点にカメラがあるので、ここでは位置はカメラ座標系そのまま）
        origin_cam = Point(x=0.0, y=0.0, z=0.0)                           # カメラ原点
        p1_cam     = Point(x= half_width, y= half_height, z= depth)         # 右上
        p2_cam     = Point(x=-half_width, y= half_height, z= depth)         # 左上
        p3_cam     = Point(x=-half_width, y=-half_height, z= depth)         # 左下
        p4_cam     = Point(x= half_width, y=-half_height, z= depth)         # 右下

        # Rodrigues 変換により rvec から回転行列 R を取得
        R, _ = cv2.Rodrigues(rvec)
        # 今回は原点固定のため、tvec の影響は無視します。
        # カメラ座標系の点を rvec で回転させるには、 p_rotated = R^T * p_cam  を用います。
        R_inv = R.T

        def rotate_point(pt_cam: Point) -> Point:
            """カメラ座標系の点 pt_cam を rvec により回転させる（原点固定）"""
            p_cam_np = np.array([pt_cam.x, pt_cam.y, pt_cam.z])
            p_rotated_np = R_inv.dot(p_cam_np)  # tvec は無視
            pt_rotated = Point()
            pt_rotated.x, pt_rotated.y, pt_rotated.z = p_rotated_np.tolist()
            return pt_rotated

        # 各点を rvec で回転させる
        origin_rot = rotate_point(origin_cam)   # 変わらず (0,0,0) になるはず
        p1_rot     = rotate_point(p1_cam)
        p2_rot     = rotate_point(p2_cam)
        p3_rot     = rotate_point(p3_cam)
        p4_rot     = rotate_point(p4_cam)

        tvec_raw = -R_inv @ tvec

        def add_point_coords(point):
            point.x += tvec_raw[0,0]
            point.y += tvec_raw[1,0]
            point.z += tvec_raw[2,0]
            return point

        origin_rot = add_point_coords(origin_rot)   # 変わらず (0,0,0) になるはず
        p1_rot     = add_point_coords(p1_rot)
        p2_rot     = add_point_coords(p2_rot)
        p3_rot     = add_point_coords(p3_rot)
        p4_rot     = add_point_coords(p4_rot)

        # Marker の作成（ここでは LINE_LIST を使用）
        marker = Marker()
        marker.header.frame_id = "map"  # 必要に応じてTFフレーム名を変更
        marker.ns = "isp"
        marker.id = 1000
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # 線分の太さ（scale.x で指定）
        marker.scale.x = 0.05

        # Marker の points リストに、以下の線分を追加
        points = []

        # カメラ原点から各四隅への線分
        points.extend([origin_rot, p1_rot])
        points.extend([origin_rot, p2_rot])
        points.extend([origin_rot, p3_rot])
        points.extend([origin_rot, p4_rot])

        # 画像平面上で四隅を結んで矩形を形成する線分
        points.extend([p1_rot, p2_rot])
        points.extend([p2_rot, p3_rot])
        points.extend([p3_rot, p4_rot])
        points.extend([p4_rot, p1_rot])

        marker.points = points

        # Marker の色を設定（例：黄色）
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker

    def publish_camera_fov_marker(self):
        if self.node.coords_converter is None:
            return
        if self.marker is None:
            self.update_marker()
        self.marker_pub.publish(self.marker)

    def update_marker(self):
        tvec = self.node.coords_converter.tvec
        rvec = self.node.coords_converter.rvec
        # tvec = np.array([[-5.07467106], [-0.17919568], [4.16808713]], dtype=np.float32)
        # rvec = np.array([[2.2486366], [-0.04807377], [0.15737083]], dtype=np.float32)
        self.node.get_logger().info(f'rvec: {rvec}, tvec: {tvec}')
        fov = (math.radians(60), math.radians(40))  # 水平・垂直画角
        depth = 0.5  # フラスタムの表示する深さ [m]
        self.marker = self.create_camera_fov_marker(tvec, rvec, depth, fov)
        self.marker.lifetime.sec = 1
