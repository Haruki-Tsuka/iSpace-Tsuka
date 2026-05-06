"""
線形最適割り当て（Linear Assignment Problem）を解くためのユーティリティ関数群。

このモジュールは、2つの配列間の最適な割り当てを見つけるための関数を提供します。
主にハンガリアン法（Hungarian algorithm）を使用して、コストを最小化する割り当てを計算します。

使用例:
    >>> a_array = np.array([1, 2, 3])
    >>> b_array = np.array([4, 5, 6])
    >>> def calc_cost(a, b): return abs(a - b)
    >>> assignments = lap_array(a_array, b_array, calc_cost)
"""

from typing import Callable, TypeVar, List
import numpy as np
import lap
from ispace_dind.data_sync.tracking.tracker import Tracker
from ispace_dind.data_model.observed_data import ObservedData
from scipy.spatial.distance import mahalanobis
T = TypeVar('T')

def lap_from_cost(cost_matrix:np.ndarray, threshold_min=-1, threshold_max=99999):
    """
    コスト行列から線形最適割り当てを計算します。

    Args:
        cost_matrix (np.ndarray): 割り当てコストを表す2次元配列
        threshold_min (float, optional): 最小コスト閾値. デフォルトは-1
        threshold_max (float, optional): 最大コスト閾値. デフォルトは99999

    Returns:
        tuple: (割り当てられた行インデックス, 割り当てられた列インデックス, コスト行列)
    """
    #線形最適割り当て
    #cost_matrixがゼロサイズの時
    if cost_matrix.size == 0:
        return (np.array([]), np.array([]), cost_matrix)
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    result = np.array([[y[idx],idx] for idx in x if threshold_min<cost_matrix[y[idx],idx]<threshold_max if idx >= 0])
    if result.shape[0] == 0:
        return (np.array([]), np.array([]), cost_matrix)
    return (result[:,0], result[:,1], cost_matrix)

def lap_from_cost2(cost_matrix:np.ndarray, threshold_min=-1, threshold_max=99999):
    """
    コスト行列から線形最適割り当てを計算します。

    Args:
        cost_matrix (np.ndarray): 割り当てコストを表す2次元配列
        threshold_min (float, optional): 最小コスト閾値. デフォルトは-1
        threshold_max (float, optional): 最大コスト閾値. デフォルトは99999

    Returns:
        tuple: (割り当てられた行インデックス, 割り当てられた列インデックス, コスト行列)
    """
    #線形最適割り当て
    #cost_matrixがゼロサイズの時
    if cost_matrix.size == 0:
        return (np.array([]), np.array([]), cost_matrix)
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    result = np.array([[y[idx],idx] for idx in x if threshold_min<cost_matrix[y[idx],idx]<threshold_max if idx >= 0])
    if result.shape[0] == 0:
        return (np.array([]), np.array([]), cost_matrix)
    return (result[:,0], result[:,1], cost)

#線形最適割り当てを行い[[a_idx, b_idx]...]のndarrayを返す
def lap_array(a_array:np.ndarray, 
              b_array:np.ndarray, 
              calc_function:Callable[[T,T], float],
              threshold_min=-1,
              threshold_max=99999):
    """
    2つの配列間の最適な割り当てを計算します。

    Args:
        a_array (np.ndarray): 最初の配列
        b_array (np.ndarray): 2番目の配列
        calc_function (Callable[[T,T], float]): 2つの要素間のコストを計算する関数
        threshold_min (float, optional): 最小コスト閾値. デフォルトは-1
        threshold_max (float, optional): 最大コスト閾値. デフォルトは99999

    Returns:
        tuple: (割り当てられたa_arrayのインデックス, 割り当てられたb_arrayのインデックス, コスト行列)
    """
    #指定された関数を用いてコスト行列を作成

    if a_array.size == 0 or b_array.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    cost_matrix = np.vectorize(calc_function)(a_array[:, np.newaxis], b_array)
    return lap_from_cost(cost_matrix, threshold_min, threshold_max)

def get_maharanobis_matrix(tracker_list:List[Tracker], observed_data_list:List[ObservedData]) -> np.ndarray:
    maharanobis_matrix = np.zeros((len(tracker_list), len(observed_data_list)))
    #tracker内のekfのx、Pを取得。ObservedData.coordとのmahalanobis_distanceを計算し、maharanobis_matrixに格納
    for tracker_idx in range(len(tracker_list)):
        tracker = tracker_list[tracker_idx]
        for observed_data_idx in range(len(observed_data_list)):
            observed_data = observed_data_list[observed_data_idx]
            maharanobis_matrix[tracker_idx, observed_data_idx] = mahalanobis_distance(tracker, observed_data)
    return maharanobis_matrix

def mahalanobis_distance(tracker:Tracker, observed_data:ObservedData) -> float:
    distance = 0.0
    if tracker.get_local_id() < 0:
        distance = np.linalg.norm(tracker.ekf.get_x()[0:3] - observed_data.coord)
    else:
        distance = mahalanobis(tracker.ekf.get_x()[0:3], observed_data.coord, tracker.ekf.get_S())
    return distance