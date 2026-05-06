import pandas as pd
import cv2
import numpy as np
import os
import pyrealsense2 as rs

from scipy.optimize import minimize


csv_data = pd.read_csv('csv/kachaka_coords.csv')
pix_data = csv_data[['pix_x', 'pix_y']]
kachaka_coords = csv_data[['kac_x', 'kac_y', 'world_z']]
coords = np.array(kachaka_coords)
pix = np.array(pix_data)

# RealSenseパイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGBカメラの設定

# ストリーム開始
pipeline.start(config)

# フレームを取得（1フレームだけ）
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# カメラの内部パラメータを取得
profile = color_frame.profile
intrinsics = profile.as_video_stream_profile().get_intrinsics()

# 内部パラメータ行列 (Intrinsic Matrix)
mtx = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]], dtype=np.float64)

dist = np.array(intrinsics.coeffs)  # 歪み係数 [k1, k2, p1, p2, k3]

# 結果を表示
print("内部パラメータ行列 K:")
print(mtx)

dist_coeffs = np.array(intrinsics.coeffs)  # 歪み係数 [k1, k2, p1, p2, k3]
print("歪み係数:", dist_coeffs)

print("カメラ内部パラメータ:")
print(f"解像度: {intrinsics.width}x{intrinsics.height}")
print(f"焦点距離 (fx, fy): {intrinsics.fx}, {intrinsics.fy}")
print(f"主点 (ppx, ppy): {intrinsics.ppx}, {intrinsics.ppy}")
print(f"歪み係数 (k1, k2, p1, p2, k3): {intrinsics.coeffs}")


# パイプライン停止
pipeline.stop()

pix = pix.astype(np.float32)
coords = coords.astype(np.float32)

retval, rvecs, tvecs = cv2.solvePnP(coords, pix, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

if retval:
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save('results/rvecs.npy', rvecs)
    np.save('results/tvecs.npy', tvecs)
    np.save('results/mtx.npy', mtx)
    np.save('results/dist.npy', dist)
    projected_points, _ = cv2.projectPoints(coords, rvecs, tvecs, mtx, dist)
    projected_points = projected_points.reshape(-1, 2)
    errors = pix - projected_points  # 各点の [Δx, Δy]
    errors_norm = np.linalg.norm(errors, axis=1)
    mean_error = np.mean(errors_norm)
    R_mat, _ = cv2.Rodrigues(rvecs)
    print("R_mat")
    print(R_mat)
    t_raw = -R_mat.T @ tvecs
    print("t_raw")
    print(t_raw)
    print('rvecs:', rvecs)
    print('tvecs:', tvecs)
    print('再投影誤差', mean_error)