import numpy as np
import cv2
import time
from utils.realsense_manager import RealSenseManager
from utils.coords_converter import CoordsConverter


class LoggingMap():

    resolution = 0.02500000037252903
    position_x = -8.19429
    position_y = -11.5681
    width = 703
    height = 770

    def __init__(self):
        self.map_img = cv2.imread("map.png")
        self.map_array = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
        self.map_data = np.zeros((self.height, self.width), dtype=np.uint8)
        self.map_data[self.map_array < 200] = 127  # 未知領域はグレー
        self.map_data[self.map_array > 245] = 255   # 空き領域は白
        self.map_data[(self.map_array >= 200) & (self.map_array <= 245)] = 0   # 障害物は黒
        self.coords = np.empty((0, 2), dtype=np.int32)
        self.timestamps = np.array([], dtype=np.float32)
        self.obj_imgs = []
        self.person_imgs = []
        pass

    def add_data(self, x, y, timestamp, obj_img, person_img):
        # ワールド座標をマップのグリッド座標に変換
        map_x, map_y = self.world_to_map(x, y)
        self.coords = np.append(self.coords, [[map_x, map_y]], axis=0)
        self.timestamps = np.append(self.timestamps, timestamp)
        self.obj_imgs.append(obj_img)
        self.person_imgs.append(person_img)

    def get_data(self, x, y, range=5):
        # self.coordsから、x-range <= x <= x+range, y-range <= y <= y+rangeの範囲のデータのインデックスを取得
        x_min = x - range
        x_max = x + range
        y_min = y - range
        y_max = y + range
        index = np.where((self.coords[:, 0] >= x_min) & (self.coords[:, 0] <= x_max) & (self.coords[:, 1] >= y_min) & (self.coords[:, 1] <= y_max))
        if len(index) == 0:
            return np.empty((0, 2), dtype=np.int32), np.array([], dtype=np.float32), [], []
        tmp_obj = [self.obj_imgs[i] for i in index[0]]
        tmp_person = [self.person_imgs[i] for i in index[0]]
        return self.coords[index], self.timestamps[index], tmp_obj, tmp_person

    def world_to_map(self, world_x, world_y):
        # マップのoriginと解像度を使ってワールド座標をマップのグリッド座標に変換
        map_x = int((world_x - self.position_x) / self.resolution)
        map_y = self.height - int((world_y - self.position_y) / self.resolution)
        return map_x, map_y
    
    def map_to_world(self, map_x, map_y):
        # マップのoriginと解像度を使ってマップのグリッド座標をワールド座標に変換
        world_x = map_x * self.resolution + self.position_x
        world_y = (self.height - map_y) * self.resolution + self.position_y
        return world_x, world_y
    
    def get_original_map(self):
        return self.map_img
    
    def print_map(self):
        img = self.map_img.copy()
        for coord in self.coords:
            cv2.circle(img, (coord[0], coord[1]), 3, (0, 0, 255), -1)
        return img

rs_manager = RealSenseManager()
log_map = LoggingMap()
map_img = log_map.get_original_map()

mtx = np.load('results/mtx.npy')
dist = np.load('results/dist.npy')
rvec = np.load('results/rvecs.npy')
tvec = np.load('results/tvecs.npy')
convertor = CoordsConverter(tvec, rvec, mtx, dist)
circle_list = []

def click_event(event, x, y, flags, param):
    global map_img
    if event == cv2.EVENT_LBUTTONDOWN:
        coords_3d = convertor.pixel2world(np.array([[x, y]]), 0)
        log_map.add_data(coords_3d[0,0], coords_3d[0,1], time.time(), rs_manager.get_img(), rs_manager.get_img())
        map_img = log_map.print_map()

def map_click_event(event, x, y, flags, param):
    global map_img
    if event == cv2.EVENT_LBUTTONDOWN:
        map_img = log_map.print_map()
        world_x, world_y = log_map.map_to_world(x, y)
        pixel_coords = convertor.world2pixel(np.array([[world_x, world_y, 0]]))
        circle_list.append((pixel_coords[0,0], pixel_coords[0,1]))

def main():
    global map_img
    while True:
        rs_manager.update()
        img = rs_manager.get_img()
        axis_length = 1  # 例として50単位

        # ワールド座標系での原点と各軸の端点を定義
        # ここでは原点、X軸、Y軸、Z軸の端点を設定しています。
        axis_points = np.float32([
            [0, 0, 0],                   # 原点
            [axis_length, 0, 0],         # X軸（赤で描画）
            [0, axis_length, 0],         # Y軸（緑で描画）
            [0, 0, axis_length]          # Z軸（青で描画）
        ])

        # ================================
        # 3D点を画像平面へ射影
        # ================================
        # cv2.projectPoints を使用して、axis_points を画像上の2D座標に変換します
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # 原点の座標
        origin = tuple(imgpts[0])

        # 各軸を画像に描画
        # X軸：赤 (BGR: (0, 0, 255))
        image = cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 3)
        # Y軸：緑 (BGR: (0, 255, 0))
        image = cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)
        # Z軸：青 (BGR: (255, 0, 0))
        image = cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)
        for circle in circle_list:
            cv2.circle(image, tuple(circle), 5, (0, 0, 255), -1)
        #imageとmap_imgを2倍に拡大
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        map_img = cv2.resize(map_img, (map_img.shape[1] * 2, map_img.shape[0] * 2))
        cv2.imshow("RealSense", image)
        cv2.setMouseCallback("RealSense", click_event)
        cv2.imshow("Map", map_img)
        cv2.setMouseCallback("Map", map_click_event)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

