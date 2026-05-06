from typing import Callable, TypeVar
import numpy as np
T = TypeVar('T')

def get_cost_matrix(a_array:np.ndarray, 
                    b_array:np.ndarray, 
                    calc_function:Callable[[T,T], float]):
    """
    2つの配列間のコスト行列を計算します。

    Args:
        a_array (np.ndarray): 最初の配列
        b_array (np.ndarray): 2番目の配列
        calc_function (Callable[[T,T], float]): 2つの要素間のコストを計算する関数

    Returns:
        np.ndarray: コスト行列
    """
    #指定された関数を用いてコスト行列を作成
    if a_array.size == 0 or b_array.size == 0:
        return np.array([])
    return np.vectorize(calc_function)(a_array[:, np.newaxis], b_array)

def softmax_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    #コスト行列が2列以上存在するかチェック
    if cost_matrix.size == 0:
        return np.array([])
    if cost_matrix.shape[1] < 2:
        return np.ones(cost_matrix.shape[0]).astype(float)
    sorted_cost = np.sort(cost_matrix, axis=1)
    margin = sorted_cost[:, 1] - sorted_cost[:, 0]
    second_min = sorted_cost[:, 1]
    relative_margin = margin / (second_min + 1e-8)
    return relative_margin
