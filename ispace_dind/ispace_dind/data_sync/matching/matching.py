import numpy as np
from ispace_dind.data_sync.assignment.assignment import lap_from_cost
from ispace_dind.data_sync.assignment.cost_matrix import softmax_cost_matrix
from ispace_dind.data_model.observed_data import ObservedData
from ispace_dind.data_model.sync_data import SyncData
from typing import List
from ispace_dind.utils.coords_converter import CoordsConverter
from ispace_dind.data_model.dind_data import DINDData
"""
座標リストを受け取り、フレーム内に存在する座標のインデックスと座標を返す。

Args:
    coord_list: 座標リスト
    frame_height: フレームの高さ
    frame_width: フレームの幅
    coords_converter: 座標変換器

Returns:
    inframe_data_idxs: フレーム内に存在する座標のインデックス
    inframe_data_array: フレーム内に存在する座標の配列
"""
def get_in_frame_data_idxs(coord_list:np.ndarray, frame_height:int, frame_width:int, coords_converter:CoordsConverter) -> np.ndarray:
    if coord_list.size == 0:
        return np.array([])
    pixel_coord = coords_converter.world2pixel(coord_list)
    # np.array([[x, y], [x, y]])のように座標が格納されているので、xとyの値をそれぞれチェックし、True/Falseのarrayを返す
    if pixel_coord.ndim == 1:
        x = np.array([pixel_coord[0]])
        y = np.array([pixel_coord[1]])
    else:
        x = pixel_coord[:, 0]
        y = pixel_coord[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)

    # 画素の包含判定（0 <= x < W, 0 <= y < H）
    in_x = (x >= 0) & (x < frame_width)
    in_y = (y >= 0) & (y < frame_height)
    bool_array = finite & in_x & in_y
    
    inframe_data_idxs = np.flatnonzero(bool_array)
    return inframe_data_idxs

def global_matching(node, dind_data:DINDData, received_data:List[SyncData], observed_data_list:List[ObservedData], frame):
    #List[SyncData]をnp.arrayに変換
    received_data_array:np.ndarray = np.asarray(received_data, dtype=object)
    observed_data_array:np.ndarray = np.asarray(observed_data_list, dtype=object)
    
    if received_data_array.size == 0 or observed_data_array.size == 0:
        return None
    
    inframe_received_data_idxs:np.ndarray = get_in_frame_data_idxs(np.array([sync_data.coord for sync_data in received_data_array]), frame.shape[0], frame.shape[1], node.coords_converter)
    inframe_observed_data_idxs:np.ndarray = get_in_frame_data_idxs(np.array([observed_data.coord for observed_data in observed_data_array]), frame.shape[0], frame.shape[1], dind_data.coords_converter)
    inframe_received_data_array:np.ndarray = received_data_array[inframe_received_data_idxs]
    inframe_observed_data_array:np.ndarray = observed_data_array[inframe_observed_data_idxs]
    if len(inframe_received_data_array) == 0 or len(inframe_observed_data_array) == 0:
        return None
    
    my_camera_pos = node.coords_converter.real_tvec.flatten() # 自分のカメラ位置を取得
    received_camera_pos = dind_data.coords_converter.real_tvec.flatten() # 他のカメラ位置を取得
    
    # コスト行列を初期化
    cost_matrix = np.zeros((len(inframe_received_data_array), len(inframe_observed_data_array)))
    cost_matrix_distance = np.zeros((len(inframe_received_data_array), len(inframe_observed_data_array)))
    mid_points = np.zeros((len(inframe_received_data_array), len(inframe_observed_data_array), 3))
    
    # 各レイ同士の最近傍点を計算
    for i, recieve_data in enumerate(inframe_received_data_array):
        # レイと取得した座標の距離を計算する（レイ付近に存在する可能性の高いデータを特定）
        received_ray = recieve_data.ray
        for j, observed_data in enumerate(inframe_observed_data_array):
            my_ray = observed_data.ray
            # 2つの直線の方向ベクトル
            d1 = my_ray - my_camera_pos
            d2 = received_ray - received_camera_pos
            
            # 最近傍点の計算に必要なパラメータ
            n = np.cross(d1, d2)
            n1 = np.cross(d1, n)
            n2 = np.cross(d2, n)
            
            # 2直線間の最短距離を計算する点
            c1 = my_camera_pos + (np.dot((received_camera_pos - my_camera_pos), n2) / np.dot(d1, n2)) * d1
            c2 = received_camera_pos + (np.dot((my_camera_pos - received_camera_pos), n1) / np.dot(d2, n1)) * d2
            
            #行列にc1,c2の中点を保存
            mid_points[i,j] = (c1 + c2) / 2
            cost_matrix_distance[i,j] = np.linalg.norm(c1-observed_data.coord) + np.linalg.norm(c2-recieve_data.coord)
            cost_matrix[i,j] = np.linalg.norm(c1 - c2)
        
    # コスト行列を最小化する割り当てを求める
    #0:default, 1:cost1, 2:cost2, 3:cost1+cost2
    if node.exp_num == 0:
        received_idxs, observed_idxs, cost_matrix = lap_from_cost(cost_matrix, threshold_min=0.0, threshold_max=0.2)
        received_distance_idxs, observed_distance_idxs, cost_matrix_distance = lap_from_cost(cost_matrix_distance, threshold_min=0.0, threshold_max=2.0)
    
        # ペア化
        pairs_cost = set(zip(received_idxs, observed_idxs))
        pairs_dist = set(zip(received_distance_idxs, observed_distance_idxs))

        # 共通するペアのみ抽出
        common_pairs = pairs_cost & pairs_dist
        
        if len(common_pairs) == 0:
            return []

        # 必要なら index 配列に戻す
        common_received_idxs = np.array([p[0] for p in common_pairs], dtype=int)
        common_observed_idxs = np.array([p[1] for p in common_pairs], dtype=int)
        
        
        #割り当ての複雑度を示す評価値を計算
        softmax_matrix = softmax_cost_matrix(cost_matrix)
        
        matching_results = []
        
        for recieved_idx, observed_idx in zip(common_received_idxs, common_observed_idxs):
            softmax_conf = softmax_matrix[recieved_idx]
            res_data = inframe_received_data_array[recieved_idx]
            
            # 観測データのインデックス, 受信データ, 確信度を追加
            if res_data.state == 0:
                matching_results.append((inframe_observed_data_idxs[observed_idx], res_data, softmax_conf, mid_points[recieved_idx, observed_idx]))
            
        return matching_results
    elif node.exp_num == 1:
        received_idxs, observed_idxs, cost_matrix = lap_from_cost(cost_matrix, threshold_min=0.0, threshold_max=0.2)
        softmax_matrix = softmax_cost_matrix(cost_matrix)
        
        matching_results = []
        
        for recieved_idx, observed_idx in zip(received_idxs, observed_idxs):
            softmax_conf = softmax_matrix[recieved_idx]
            res_data = inframe_received_data_array[recieved_idx]
            
            # 観測データのインデックス, 受信データ, 確信度を追加
            if res_data.state == 0:
                matching_results.append((inframe_observed_data_idxs[observed_idx], res_data, softmax_conf, mid_points[recieved_idx, observed_idx]))
            
        return matching_results
    elif node.exp_num == 2:
        received_distance_idxs, observed_distance_idxs, cost_matrix_distance = lap_from_cost(cost_matrix_distance, threshold_min=0.0, threshold_max=2.0)
        softmax_matrix = softmax_cost_matrix(cost_matrix_distance)
        
        matching_results = []
        
        for recieved_idx, observed_idx in zip(received_distance_idxs, observed_distance_idxs):
            softmax_conf = softmax_matrix[recieved_idx]
            res_data = inframe_received_data_array[recieved_idx]
            if res_data.state == 0:
                matching_results.append((inframe_observed_data_idxs[observed_idx], res_data, softmax_conf, mid_points[recieved_idx, observed_idx]))
            
        return matching_results
    
    elif node.exp_num == 3:
        cost_matrix = cost_matrix + cost_matrix_distance
        received_idxs, observed_idxs, cost_matrix = lap_from_cost(cost_matrix, threshold_min=0.0, threshold_max=2.2)
        softmax_matrix = softmax_cost_matrix(cost_matrix)
        
        matching_results = []
        
        for recieved_idx, observed_idx in zip(received_idxs, observed_idxs):
            softmax_conf = softmax_matrix[recieved_idx]
            res_data = inframe_received_data_array[recieved_idx]
            
            if res_data.state == 0:
                matching_results.append((inframe_observed_data_idxs[observed_idx], res_data, softmax_conf, mid_points[recieved_idx, observed_idx]))
            
        return matching_results

def face_matching(node, dind_data:DINDData, received_data:List[SyncData], observed_data_list:List[ObservedData], frame):
    #List[SyncData]をnp.arrayに変換
    received_data_array:np.ndarray = np.asarray(received_data, dtype=object)
    observed_data_array:np.ndarray = np.asarray(observed_data_list, dtype=object)
    
    if received_data_array.size == 0 or observed_data_array.size == 0:
        return None
    
    inframe_received_data_idxs:np.ndarray = get_in_frame_data_idxs(np.array([sync_data.coord for sync_data in received_data_array]), frame.shape[0], frame.shape[1], node.coords_converter)
    inframe_received_data_array:np.ndarray = received_data_array[inframe_received_data_idxs]
    inframe_observed_data_array:np.ndarray = observed_data_array
    if len(inframe_received_data_array) == 0 or len(inframe_observed_data_array) == 0:
        return None
    
    my_camera_pos = node.coords_converter.real_tvec.flatten() # 自分のカメラ位置を取得
    received_camera_pos = dind_data.coords_converter.real_tvec.flatten() # 他のカメラ位置を取得
    
    # コスト行列を初期化
    cost_matrix = np.zeros((len(inframe_received_data_array), len(inframe_observed_data_array)))
    mid_points = np.zeros((len(inframe_received_data_array), len(inframe_observed_data_array), 3))
    
    # 各レイ同士の最近傍点を計算
    for i, recieve_data in enumerate(inframe_received_data_array):
        # レイと取得した座標の距離を計算する（レイ付近に存在する可能性の高いデータを特定）
        received_ray = recieve_data.ray
        for j, observed_data in enumerate(inframe_observed_data_array):
            my_ray = observed_data.ray
            # 2つの直線の方向ベクトル
            d1 = my_ray - my_camera_pos
            d2 = received_ray - received_camera_pos
            
            # 最近傍点の計算に必要なパラメータ
            n = np.cross(d1, d2)
            n1 = np.cross(d1, n)
            n2 = np.cross(d2, n)
            
            # 2直線間の最短距離を計算する点
            c1 = my_camera_pos + (np.dot((received_camera_pos - my_camera_pos), n2) / np.dot(d1, n2)) * d1
            c2 = received_camera_pos + (np.dot((my_camera_pos - received_camera_pos), n1) / np.dot(d2, n1)) * d2
            
            #行列にc1,c2の中点を保存
            mid_points[i,j] = (c1 + c2) / 2
            cost_matrix[i,j] = np.linalg.norm(c1 - c2)
        
    # コスト行列を最小化する割り当てを求める
    received_idxs, observed_idxs, cost_matrix = lap_from_cost(cost_matrix, threshold_min=0.0, threshold_max=0.2)
    
    softmax_matrix = softmax_cost_matrix(cost_matrix)

    matching_results = []
    for recieved_idx, observed_idx in zip(received_idxs, observed_idxs):
        softmax_conf = softmax_matrix[recieved_idx]
        res_data = inframe_received_data_array[recieved_idx]
        
        # 受信データ, 確信度を追加
        if res_data.state == 0:
            matching_results.append((observed_data_array[observed_idx], res_data, softmax_conf, mid_points[recieved_idx, observed_idx]))
    
    return matching_results