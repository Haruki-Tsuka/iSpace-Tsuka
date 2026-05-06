from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

class Observer(ABC):
    """
    観測を行う基底クラス。
    異なるカメラやデータソースからの観測を統一的に扱うためのインターフェースを提供します。
    """

    def __init__(self, node):
        """
        Args:
            node: ROS2ノードインスタンス
        """
        self.node = node
        
    @abstractmethod
    def observe(self):
        """
        観測を実行し、結果を返します。
        このメソッドはサブクラスで実装する必要があります。

        Returns:
            Tuple[NDArray, NDArray, dict]: 
                - 元画像
                - 処理済み画像
                - 観測データ（座標、重み、距離など）
        """
        pass

    def get_yolo_data(self, results) -> Tuple[bool, Optional[NDArray], Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
        """
        YOLOの検出結果から必要なデータを抽出します。

        Args:
            results: YOLOの検出結果 (poseの結果)

        Returns:
            Tuple[bool, Optional[NDArray], Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
                - 成功フラグ: データの抽出が成功したかどうか
                - バウンディングボックス座標: [N, 4]の配列 (x1, y1, x2, y2)
                - 信頼度スコア: [N]の配列
                - キーポイント座標: [N, 17, 2]の配列
                - キーポイント信頼度: [N, 17]の配列
        """
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
    
    def get_iou_array(self, bbox_xyxy: NDArray) -> NDArray:
        """
        バウンディングボックス間の不一致率（1-IoU）を計算します。

        Args:
            bbox_xyxy: バウンディングボックスの座標配列 [N, 4] (x1, y1, x2, y2)

        Returns:
            NDArray: 各バウンディングボックスに対する不一致率の配列 [N]
        """
        b_area = (bbox_xyxy[:,2] - bbox_xyxy[:,0] + 1) \
             * (bbox_xyxy[:,3] - bbox_xyxy[:,1] + 1)
        iou_array = np.zeros(bbox_xyxy.shape[0])
        for i in range(bbox_xyxy.shape[0]):
            abx_mn = np.maximum(bbox_xyxy[i,0], bbox_xyxy[:,0]) # xmin
            aby_mn = np.maximum(bbox_xyxy[i,1], bbox_xyxy[:,1]) # ymin
            abx_mx = np.minimum(bbox_xyxy[i,2], bbox_xyxy[:,2]) # xmax
            aby_mx = np.minimum(bbox_xyxy[i,3], bbox_xyxy[:,3]) # ymax
            w = np.maximum(0, abx_mx - abx_mn + 1)
            h = np.maximum(0, aby_mx - aby_mn + 1)
            intersect = w*h
            iou = intersect / (b_area[i]+b_area-intersect)
            iou[i] = 0
            iou_array[i] = 1-np.max(iou)
        return iou_array
    
    def get_kp_weight(self, kp_conf: NDArray) -> NDArray:
        """
        キーポイントの信頼度から重みを計算します。
        各キーポイントペア（左右）の最大信頼度の平均を計算します。

        Args:
            kp_conf: キーポイントの信頼度配列 [N, 17]

        Returns:
            NDArray: 各人物に対する重みの配列 [N]
        """
        means = np.zeros(kp_conf.shape[0])
        for i in range(kp_conf.shape[0]):
            means[i] = np.mean([max(kp_conf[i,2*j-1], kp_conf[i,2*j]) for j in range(1,9)])
        return means
    
    
    def get_nose_code(self, kp: NDArray) -> NDArray:
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
    
    def nose_filter(self, nose_coords: NDArray, kp_conf: NDArray, threshold: float = 1) -> NDArray:
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