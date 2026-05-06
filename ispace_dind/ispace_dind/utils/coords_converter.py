import cv2
import numpy as np

class CoordsConverter:

    '''
    tvec = 並進ベクトル(3, 1)
    rvec = 回転ベクトル(3, 1)
    matrix = カメラ行列(3, 3)
    dist = 歪み係数(1, 5)
    '''
    def __init__(self, tvec:np.ndarray, rvec:np.ndarray, camera_matrix:np.ndarray, dist_coeffs:np.ndarray):
        self.tvec = tvec
        self.rvec = rvec
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.R, _ = cv2.Rodrigues(rvec)
        self.R_inv = self.R.T
        self.flatten_tvec = tvec.flatten()
        self.real_tvec = -self.R_inv @ self.tvec

    def get_real_tvec(self):
        return self.real_tvec
    
    def get_rvec(self):
        return self.rvec

    '''
    3次元座標をworld座標系に変換
    3次元座標の形状は(N, 3)である必要がある(N=1: np.array([[x,y,z]]))
    '''
    def camera2world(self, coords_3d:np.ndarray):
        world_points = (self.R.T @ (coords_3d.T - self.tvec)).T
        return world_points

    '''
    2次元座標をworld座標系に変換
    2次元座標の形状は(N, 2)である必要がある(N=1: np.array([[u,v]]))
    '''
    def pixel2world(self, pixel_coords:np.ndarray, world_z:float):
        if pixel_coords.size == 0:
            return np.array([[]])
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords[np.newaxis, :]
        pts = pixel_coords.reshape(-1, 1, 2).astype(np.float64)
        normalized_pts = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, None, None).reshape(-1, 2)
        N = normalized_pts.shape[0]
        p = np.hstack((normalized_pts, np.ones((N, 1), dtype=np.float64)))  # shape: (N,3)
        RT_z = self.R_inv[2, :]
        denom = np.dot(p, RT_z) 
        constant = np.dot(RT_z, self.flatten_tvec)
        s = (world_z + constant) / denom
        P_cam = p * s[:, np.newaxis]
        P_world = np.dot(P_cam - self.flatten_tvec, self.R_inv.T)
        return P_world
    
    '''
    world座標系を2次元座標に変換
    3次元座標の形状は(N, 3)である必要がある(N=1: np.array([[x,y,z]]))
    '''
    def world2pixel(self, world_coords:np.ndarray):
        world_coords = np.asarray(world_coords, dtype=np.float64)
        if world_coords.size == 0:
            return np.array([])
        if world_coords.ndim == 1:
            world_coords = world_coords[np.newaxis, :]
        points = cv2.projectPoints(world_coords, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        return np.int32(points[0]).reshape(-1, 2)