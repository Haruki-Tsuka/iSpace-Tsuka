from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import mahalanobis
import numpy as np
import copy
from ispace_dind.ros_bridge.message_utils import ros_now_sec

class EKF3D(ExtendedKalmanFilter):
    
    '''
    3DのEKFを管理するクラス
    initial_z: 初期観測値 [x, y, z]
    ekf.x: 状態ベクトル [x, y, z, vx, vy, vz]
    '''
    def __init__(self, initial_z: np.ndarray, range_std: float=0.1):
        super().__init__(dim_x=6, dim_z=3)
        self.x = np.array([initial_z[0], initial_z[1], initial_z[2], 0, 0, 0])
        self.R = np.diag([range_std, range_std, range_std*2])
        self.Q = self.__make_Q(dt=0.1, sigma_a=2.0)
        self.P *= 0.5
        self.last_update = ros_now_sec()
        self.last_predict = ros_now_sec()
        self.last_update_ekf = None
        self.dt_list = np.array([])
        
    def __get_f(self, dt):
        return np.eye(6)+np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])*dt
    
    def __get_h_jacobian(self, x):
        return np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])

    def __get_h_x(self, x):
        return np.array([x[0], x[1], x[2]]).T
    
    def __make_Q(self, dt, sigma_a):
        q = sigma_a**2 * np.array([[dt**4/4, dt**3/2],
                            [dt**3/2, dt**2]])
        Q = np.zeros((6,6))
        for i in range(3):
            Q[[i,i+3],[i,i+3]] = q.diagonal()
            Q[i, i+3] = Q[i+3, i] = q[0,1]
        return Q
    
    def predict(self):
        dt = ros_now_sec() - max(self.last_predict, self.last_update)
        if self.dt_list.size == 0:
            self.last_update_ekf = copy.deepcopy(self.ekf)
        self.F = self.__get_f(dt)
        super().predict()
        self.last_predict = ros_now_sec()
        self.dt_list = np.append(self.dt_list, dt)
        
    def update(self, z):
        if self.dt_list.size > 1:
            self.oru(z, method='bezier')
        else:
            super().update(z, Hx=self.__get_h_x, HJacobian=self.__get_h_jacobian)
        self.last_update = ros_now_sec()
        self.dt_list = np.array([])
    
    def oru(self, z, method='bezier'):
        if method == 'bezier':
            for i in range(self.dt_list.size):
                self.last_update_ekf.predict()
                ratio = np.sum(self.dt_list[0:i]) / np.sum(self.dt_list)
                p_0 = self.last_update_ekf.x[0:3]
                p_1 = self.last_update_ekf.x[0:3] + self.dt_list[i] * self.last_update_ekf.x[3:6]
                p_2 = z
                estimate_coord = (1-ratio)**2 * p_0 + 2 * (1-ratio) * ratio * p_1 + ratio**2 * p_2
                self.last_update_ekf.F = self.__get_f(self.dt_list[i])
                self.last_update_ekf.update(estimate_coord, Hx=self.__get_h_x, HJacobian=self.__get_h_jacobian)
            self = self.last_update_ekf
            
    def get_x(self):
        return self.ekf.x
    
    def get_P(self):
        return self.ekf.P
    
    def get_S(self):
        return self.ekf.S
    
    def get_predicted_seconds(self):
        return self.dt_list.sum()