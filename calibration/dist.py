import cv2
import numpy as np
from utils.realsense_manager import RealSenseManager

rs_manager = RealSenseManager(640, 480, 30)

K = np.load('results/mtx.npy')
distCoeffs = np.load('results/dist.npy')

K = np.array([[524.82282383, 0, 402.38954688],
              [0, 537.90728237, 237.6349161],
              [0, 0, 1]], dtype=np.float64)

distCoeffs = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)

while True:
    rs_manager.update()
    img = rs_manager.get_img()
    # 画像サイズの取得
    h, w = img.shape[:2]

    # 最適な新しいカメラ行列の計算
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
    # ゆがみ補正（Undistortion）
    undistorted_img = cv2.undistort(img, K, distCoeffs, None, newCameraMatrix)
    cv2.imshow('frame', undistorted_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(newCameraMatrix)
        break