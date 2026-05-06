from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import mahalanobis
import numpy as np
from rclpy.clock import Clock, ClockType
import copy
from ispace_dind.ros_bridge.message_utils import ros_now_sec

def HJacobian_at(x):
    #引数のxは、今回は使用しないけど必要
    return np.array([[1,0,0,0],[0,1,0,0]])

def Hx(x):
    return np.array([x[0],x[1]]).T

def HJacobian_at2(x):
    #引数のxは、今回は使用しないけど必要
    return np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]])

def Hx2(x):
    return np.array([x[0],x[1],x[2]]).T


class EKF_2D:

    next_id = 1

    def __init__(self, initial_coord:np.ndarray=np.array([0,0,0,0]), range_std:float=0.1):
        #状態が[x,y,vx,vy]、観測は[x,y]、初期状態は[0,0,0,0]
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        self.range_std = range_std

        self.ekf.x = initial_coord
        self.ekf.R =range_std
        self.ekf.Q[0:2,0:2] = Q_discrete_white_noise(2, dt=1, var=0.1)
        self.ekf.Q[2,2] = 0.1
        self.ekf.P *= 0.5
        self.last_update = ros_now_sec()
        self.last_predict = ros_now_sec()
        self.pred_P = np.zeros((4,4))
        self.pred_times = 0

    #外部からのデータ更新時、last_updateとlast_predictを最新にしないといけない

    def __get_f(self, dt):
        return np.eye(4)+np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])*dt

    def predict(self):
        dt = ros_now_sec() - max(self.last_predict, self.last_update)
        self.ekf.F = self.__get_f(dt)
        self.ekf.predict()
        self.pred_P = copy.deepcopy(self.ekf.P)
        self.last_predict = ros_now_sec()

    def update(self, z, dt):
        self.ekf.update(z, HJacobian=HJacobian_at, Hx=Hx)
        self.last_update = ros_now_sec()

    def get_coord(self) -> np.ndarray:
        return self.ekf.x[0:2]

    def mahalanobis_distance(self, coord):
        return mahalanobis(x=coord, mean=self.ekf.x[0:2], cov=self.ekf.P[0:2,0:2]) + (self.pred_times-1)*self.range_std
    
    def get_time_diff(self):
        return ros_now_sec() - self.last_update
    
class KalmanPersonTracker(EKF_2D):
    next_id = 1
    next_propose_id = -1
    def __init__(self, initial_coord: np.ndarray=np.array([0,0,0,0]), range_std: float=0.1, life_time=1.5, timestamp=None):
        super().__init__(initial_coord, range_std)
        self.life_time = life_time
        self.tracker_id = KalmanPersonTracker.next_propose_id
        KalmanPersonTracker.next_propose_id -= 1
        self.global_id = -1
        self.is_pred = False
        self.ray = None
        self.nose_code = None
        self.conf = 0.0
        self.inherited_data = ''
        self.inherited_dind_id = ''
        self.z = 0.0
        self.assosiate_times = 0
        self.proposed_id = ''
        
        self.create_id = None
        self.create_time = None
        
        self.bbox = None
        self.keypoints = None
        
    def set_ray(self, ray):
        self.ray = ray

    def predict(self) -> bool:
        #一定時間更新の無い場合、誤差拡大防止の為推定を行わない
        if self.__now() - self.last_update > self.life_time:
            return False
        #生成されて、次のフレームで更新の無いデータは削除
        if self.is_pred and self.tracker_id <= -1:
            return False
        super().predict()
        self.is_pred = True
        self.pred_times += 1
        print('predict', self.tracker_id)
        return True
    
    def update(self, z, timestamp):
        #IDを付与
        if self.tracker_id <= -1:
            self.tracker_id = KalmanPersonTracker.next_id   
            KalmanPersonTracker.next_id += 1
            self.create_time = timestamp
            self.create_id += f'_{self.tracker_id}'
        self.z = z[2]
        self.assosiate_times += 1
        self.pred_times = 0
        super().update(z[0:2], timestamp - max(self.last_update, self.last_predict))
        print('update', self.tracker_id)
    
    def set(self, mat_x, mat_p, mat_k):
        self.ekf.x = mat_x
        self.ekf.P = mat_p
        self.ekf.K = mat_k
        self.last_update = self.__now()

    def __now(self):
        return Clock(clock_type=ClockType.ROS_TIME).now().nanoseconds / 1000000000

class EKF_3D:

    next_id = 1

    def __init__(self, point_data, range_std:float=0.1):
        #状態が[x,y,z,vx,vy]、観測は[x,y,z]、初期状態は[0,0,0,0,0]
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=3)
        self.range_std = range_std

        self.ekf.x = np.array([point_data.get_coord()[0], point_data.get_coord()[1], point_data.get_coord()[2], 0, 0])
        self.ekf.R = range_std
        self.ekf.Q[0:3,0:3] = Q_discrete_white_noise(3, dt=1, var=0.1)
        self.ekf.P *= 0.5
        self.last_update = ros_now_sec()
        self.last_predict = ros_now_sec()
        self.last_update_ekf = None
        self.dt_list = np.array([])
        self.pred_times = 0

    def __get_f(self, dt):
        return np.eye(5)+np.array([[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])*dt

    def predict(self):
        dt = ros_now_sec() - max(self.last_predict, self.last_update)
        if self.dt_list.size == 0:
            self.last_update_ekf = copy.deepcopy(self.ekf)
        self.ekf.F = self.__get_f(dt)
        self.ekf.predict()
        self.pred_P = copy.deepcopy(self.ekf.P)
        self.last_predict = ros_now_sec()
        self.dt_list = np.append(self.dt_list, dt)
        
    def update(self, z):
        if self.dt_list.size > 1:
            self.oru(z, method='bezier')
        else:
            self.ekf.update(z, HJacobian=HJacobian_at2, Hx=Hx2)
        self.last_update = ros_now_sec()
        self.dt_list = np.array([])
        
    def oru(self, z, method='bezier'):
        if method == 'bezier':
            for i in range(self.dt_list.size):
                #アップデートが無かった箇所をベジエ曲線で補完し、過去に戻ってアップデートを行う
                #OC-SORTのORUを線形からベジェ曲線に変更（自然な経路になる可能性）
                ratio = np.sum(self.dt_list[0:i]) / np.sum(self.dt_list)
                p_0 = self.last_update_ekf.x[0:2]
                p_1 = self.last_update_ekf.x[0:2] + self.dt_list[i] * self.last_update_ekf.x[3:5]
                p_2 = z[0:2]
                estimate_coord = (1-ratio)**2 * p_0 + 2 * (1-ratio) * ratio * p_1 + ratio**2 * p_2
                estimate_coord = np.array([estimate_coord[0], estimate_coord[1], z[2]])
                self.last_update_ekf.F = self.__get_f(self.dt_list[i])
                print('estimate', estimate_coord)
                self.last_update_ekf.update(estimate_coord, HJacobian=HJacobian_at2, Hx=Hx2)
            self.ekf = self.last_update_ekf
        elif method == 'linear':
            for i in range(self.dt_list.size):
                ratio = np.sum(self.dt_list[0:i]) / np.sum(self.dt_list)
                p_0 = self.last_update_ekf.x[0:2]
                p_1 = z[0:2]
                estimate_coord = (1-ratio) * p_0 + ratio * p_1
                estimate_coord = np.array([estimate_coord[0], estimate_coord[1], z[2]])
                self.last_update_ekf.F = self.__get_f(self.dt_list[i])
                print('estimate', estimate_coord)
                self.last_update_ekf.update(estimate_coord, HJacobian=HJacobian_at2, Hx=Hx2)
            self.ekf = self.last_update_ekf

    def get_coord(self) -> np.ndarray:
        return self.ekf.x[0:3]
    
    def get_cov(self):
        return self.ekf.P[0:3,0:3]

    def mahalanobis_distance(self, coord):
        return mahalanobis(x=coord, mean=self.ekf.x[0:3], cov=self.ekf.P[0:3,0:3]) + (self.pred_times-1)*self.range_std
    
    def get_time_diff(self):
        return ros_now_sec() - self.last_update
    
class NewKalmanPersonTracker(EKF_3D):
    next_id = 1
    next_propose_id = -1
    def __init__(self, point_data, range_std: float=0.1, life_time=1.5, timestamp=None):
        super().__init__(point_data, range_std)
        self.life_time = life_time
        self.tracker_id = NewKalmanPersonTracker.next_propose_id
        NewKalmanPersonTracker.next_propose_id -= 1
        self.is_pred = False
        
        self.point_data = point_data

        self.inherited_data = ''
        self.inherited_dind_id = ''
        self.assosiate_times = 1
        self.proposed_id = ''
        
        self.create_id = None
        self.create_time = None
        
    def predict(self) -> bool:
        #一定時間更新の無い場合、誤差拡大防止の為推定を行わない
        if ros_now_sec() - self.last_update > self.life_time:
            return False
        #生成されて、次のフレームで更新の無いデータは削除
        if self.is_pred and self.tracker_id <= -1:
            return False
        super().predict()
        self.is_pred = True
        self.pred_times += 1
        print('predict', self.tracker_id)
        return True
    
    def update(self, point_data, timestamp):
        #IDを付与
        if self.tracker_id <= -1:
            self.tracker_id = NewKalmanPersonTracker.next_id   
            NewKalmanPersonTracker.next_id += 1
            self.create_time = timestamp
            self.create_id += f'_{self.tracker_id}'
        self.point_data = point_data
        self.assosiate_times += 1
        self.pred_times = 0
        super().update(point_data.get_coord())
        print('update', self.tracker_id)
    
    def set(self, mat_x, mat_p, mat_k):
        self.ekf.x = mat_x
        self.ekf.P = mat_p
        self.ekf.K = mat_k
        self.last_update = ros_now_sec()
    
    def get_point_data(self):
        return self.point_data
    
    def get_ray(self):
        return self.point_data.get_ray()
    
    def get_bbox(self):
        return self.point_data.get_bbox()
    
    def get_keypoints(self):
        return self.point_data.get_keypoints()
    
    def get_conf(self):
        return self.point_data.get_conf()
    
    def get_nose(self):
        return self.point_data.get_nose()