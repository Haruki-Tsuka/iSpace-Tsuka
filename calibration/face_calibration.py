import cv2
import numpy as np
import os
from utils.coords_converter import CoordsConverter
from utils.realsense_manager import RealSenseManager
from ultralytics import YOLO
from typing import Tuple
import time
import json
import socket
import struct

def send_msg(sock: socket.socket, payload: bytes) -> None:
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)

def get_coords_converter():
    if os.path.exists('camera_matrix'):
        rvec = np.load('camera_matrix/rvecs.npy')
        tvec = np.load('camera_matrix/tvecs.npy')
        mtx = np.load('camera_matrix/mtx.npy')
        dist = np.load('camera_matrix/dist.npy')
    else:
        os.makedirs('camera_matrix')
        raise FileNotFoundError('カメラ行列が存在しません。')
    return CoordsConverter(tvec, rvec, mtx, dist)

def get_nose_code(kp: np.ndarray) -> np.ndarray:
        """
        キーポイントから鼻の座標を取得します。
        顔の上部5点の平均位置を鼻の位置として使用します。

        Args:
            kp: キーポイント座標配列 [N, 17, 2]

        Returns:
            NDArray: 鼻の座標配列 [N, 2]。座標が取得できない場合は[0,0]
        """
        nose_coords = []
        for kp_xy in kp:
            mask = kp_xy[:5,0] != 0
            if np.any(mask):
                ex = int(np.mean(kp_xy[:5,0][mask]))
                ey = int(np.mean(kp_xy[:5,1][mask]))
            else:
                ex, ey = 0, 0
            nose_coords.append([ex,ey])
        return np.array(nose_coords)
    
def nose_filter(nose_coords: np.ndarray, kp_conf: np.ndarray, threshold: float = 1) -> np.ndarray:
    """
    近接する鼻の座標をフィルタリングします。
    指定された閾値以下の距離にある座標ペアについて、
    信頼度の低い方を[0,0]に設定します。

    Args:
        nose_coords: 鼻の座標配列 [N, 2]
        kp_conf: キーポイントの信頼度配列 [N, 17]
        threshold: フィルタリングの距離閾値（デフォルト: 11）

    Returns:
        NDArray: フィルタリング後の鼻の座標配列 [N, 2]
    """
    #座標間の距離がthreshold以下の場合、kp_confの総和が小さい方を[0,0]にする
    for i in range(len(nose_coords)):
        for j in range(i+1, len(nose_coords)):
            if np.linalg.norm(nose_coords[i] - nose_coords[j]) < threshold:
                if np.sum(kp_conf[i]) < np.sum(kp_conf[j]):
                    nose_coords[i] = [0,0]
                else:
                    nose_coords[j] = [0,0]
    return nose_coords

def get_yolo_data(results) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if results is None:
        return False, None, None, None, None
    if results[0].boxes.conf is None:
        return False, None, None, None, None
    if results[0].keypoints.conf is None:
        return False, None, None, None, None
    kp_conf = results[0].keypoints.conf.detach().cpu().numpy()
    kp = results[0].keypoints.xy.detach().cpu().numpy()
    #kp[i,j]=[0,0]の場合、kp_conf[i]も0にする(numpyの仕様)
    kp_conf = kp_conf * (kp != 0).all(axis=2)
    return True, results[0].boxes.xyxy.detach().cpu().numpy(), results[0].boxes.conf.detach().cpu().numpy(), kp, kp_conf
    
def get_3d_coords(pixel_coords: np.ndarray, depth: np.ndarray, coords_converter: CoordsConverter, area: int = 5, limit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        2Dピクセル座標と深度画像から3D座標を計算します。
        指定されたピクセル周辺の深度値の中央値を使用して3D座標を計算します。

        Args:
            pixel_coords: 2Dピクセル座標配列 [N, 2]
            depth: 深度画像 [H, W]
            area: 深度計算のための周辺領域サイズ（デフォルト: 5）
            limit: 画像端からの最小距離（デフォルト: 10）

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D座標配列 [M, 3]。Mは有効な座標の数（N以下）
        """
        coords_3d = []
        select_idxs = []
        for i in range(pixel_coords.shape[0]):
            x, y = pixel_coords[i]
            if x < limit or x >= depth.shape[1]-limit:
                continue
            #ｘ−エリアからｘ＋エリア、ｙ−エリアからｙ＋エリアまでの深度の中央値を求める。ただし、範囲が領域外に出ないように
            x_min = max(area, x - area)
            x_max = min(depth.shape[1]-area, x + area)
            y_min = max(area, y - area)
            y_max = min(depth.shape[0]-area, y + area)
            depth_area = depth[y_min:y_max, x_min:x_max]
            nonzero_depth = depth_area[depth_area != 0]
            # Calculate the median of the non-zero values
            median_depth = np.median(nonzero_depth)
            coord_3d = coords_converter.pixel2world(np.array([x, y]), median_depth)
            if coord_3d is None:
                continue
            else:
                select_idxs.append(i)
                coords_3d.append(coord_3d)
        return np.array(coords_3d, dtype=np.float32), select_idxs
    
def main():
    coords_converter = get_coords_converter()
    rs_manager = RealSenseManager()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('192.168.1.112', 55580))
        print("[client] connected")
    yolo = YOLO("yolo11m-pose.engine")
    while True:
        rs_manager.update()
        frame = rs_manager.get_img()
        results = yolo(frame, verbose=False, conf=0.5)
        data = get_yolo_data(results)
        if not data[0]:
            continue
        has_data, bbox, bbox_conf, kp, kp_conf = data
        nose_coords = get_nose_code(kp)
        nose_coords = nose_filter(nose_coords, kp_conf, threshold=11)
        coords_3d, select_idxs = get_3d_coords(nose_coords, rs_manager.get_depth_numpy(), coords_converter, area=5, limit=10)
        if coords_3d.shape[0] == 0:
       	    continue
       	print(coords_3d)
        msg = {
            "head_xyz": f'{[coords_3d[0,0,0], coords_3d[0,0,1], coords_3d[0,0,2]]}',
            "ts": time.time()
        }
        send_msg(sock, json.dumps(msg).encode('utf-8'))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
