import os
import numpy as np
import cv2
import sys
import glob
import socket
import argparse
import threading
import importlib.util
from typing import List, Any

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ispace_dind.utils.coords_converter import CoordsConverter
from ispace_dind.utils.event_handler import EventHandler, Event
from ispace_dind.utils.config import YamlConfig
from ispace_dind.data_model.dind_data import DINDData
from ispace_dind.ros_bridge.ros_interface import ROSInterface
from ispace_dind.observer.camera.realsense import RealsenseCamera
from ispace_dind.observer.camera.dataset_camera import DatasetCamera
from ispace_dind.observer.observer_3d import Observer3D
from ispace_dind.observer.yolo_obs import ObserverYolo
from ispace_dind.observer.camera.face_camera import FaceCamera
from ispace_dind.observer.observer_face import ObserverFace
from ispace_dind.observer.camera.camera_base import CameraBase
from ispace_dind.data_sync.improved_sync import ImprovedRaySync
from ispace_dind.data_sync.face_sync import FaceSync

class DIND(Node):
    """
    分散制御によるトラッキングを行うメインクラス。
    観測、データ同期、座標変換などの機能を統合します。

    Attributes:
        hostname (str): ノードのホスト名
        coords_converter (CoordsConverter): 座標変換を行うインスタンス
        observer (Any): 観測を行うインスタンス
        data_sync (Any): データ同期を行うインスタンス
        event_handler (EventHandler): イベント管理を行うインスタンス
    """

    def __init__(self):
        """
        DINDノードを初期化します。
        ホスト名に基づいてノード名を設定し、必要なコンポーネントを初期化します。
        """
        super().__init__(f'dind_{socket.gethostname()}')
        self.hostname = socket.gethostname()
        self.frame_width = 640
        self.frame_height = 480
        self.visualize = True
        
        #TODO カメラ関連のリファクタリング
        self.camera: CameraBase = None
        self.camera_img: np.ndarray = None
        
        self.coords_converter: CoordsConverter = None
        self.dind_data_dict : dict[str, DINDData] = {}
        self.observer = None
        self.data_sync = None
        self.event_handler = EventHandler()
        self.executor_thread = threading.Thread(target=self.loop_runner, daemon=True)
        self.executor_thread.start()
        self.get_logger().info('DIND has created!')
        self.ros_interface = ROSInterface(self)
        self.config_dict: dict[str, Any] = {}
        
        self.exp_num = 0

    def loop_runner(self):
        """
        メインループを実行します。
        観測とデータ同期を定期的に実行します。
        また、他のDINDからの初期データ受信のため、開始前にクールタイムを設けています。
        """
        cooltime = 50
        while True:
            if self.camera is None:
                continue
            self.camera.update()
            self.camera_img = self.camera.get_img()
            observed_data_list = None
            if cooltime < 0:
                if self.observer is not None:
                    observed_data_list = self.observer.observe()
                if self.data_sync is not None:
                    self.data_sync.assosiate_data(observed_data_list)
            else:
                cooltime -= 1
            if self.visualize:
                cv2.imshow('dind_output', self.camera_img)
                cv2.setMouseCallback('dind_output', self.clicked)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.get_logger().info('DIND has finished!')
                    break
            
    def clicked(self, event, x, y, flags, param):
        self.event_handler.emit(Event.CLICKED_EVENT, event, x, y, flags, param)
                
    def set_camera(self, camera: CameraBase) -> None:
        """
        カメラインスタンスを設定します。
        """
        self.camera = camera
    
    def set_coords_converter(self, coords_converter: CoordsConverter) -> None:
        """
        座標変換インスタンスを設定します。

        Args:
            coords_converter: 座標変換を行うインスタンス
        """
        self.coords_converter = coords_converter

    def set_observer(self, observer: Any) -> None:
        """
        観測インスタンスを設定します。

        Args:
            observer: 観測を行うインスタンス
        """
        self.observer = observer

    def set_data_sync(self, data_sync: Any) -> None:
        """
        データ同期インスタンスを設定します。

        Args:
            data_sync: データ同期を行うインスタンス
        """
        self.data_sync = data_sync
    
    def get_data_sync(self) -> Any:
        """
        データ同期インスタンスを取得します。

        Returns:
            データ同期を行うインスタンス
        """
        return self.data_sync

    def get_coords_converter(self) -> CoordsConverter:
        """
        座標変換インスタンスを取得します。

        Returns:
            座標変換を行うインスタンス
        """
        return self.coords_converter
    
    def create_multi_threaded_timer(self, period: float, callback: callable) -> Any:
        """
        マルチスレッドタイマーを作成します。

        Args:
            period: タイマーの周期（秒）
            callback: 実行するコールバック関数

        Returns:
            作成されたタイマー
        """
        return self.create_timer(period, callback, callback_group=MutuallyExclusiveCallbackGroup())
    
def load_addons(node: DIND, addons_dir: str = 'ispace_dind/ispace_dind/addons', exclude: List[str] = [], include: List[str] = []) -> List[Any]:
    """
    アドオンモジュールを動的にロードします。

    Args:
        node: DINDノードインスタンス
        addons_dir: アドオンディレクトリのパス
        exclude: 除外するアドオンファイルのリスト

    Returns:
        ロードされたアドオンのリスト
    """
    addons = []
    if not os.path.exists(addons_dir):
        node.get_logger().warn(f'アドオンディレクトリ {addons_dir} が存在しません。')
        return addons

    for addon_file in glob.glob(os.path.join(addons_dir, '*.py')):
        if os.path.basename(addon_file) == '__init__.py' or os.path.basename(addon_file) == 'addon_base.py' or os.path.basename(addon_file) in exclude or os.path.basename(addon_file) not in include:
            continue
        module_name = os.path.splitext(os.path.basename(addon_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, addon_file)
        addon_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(addon_module)
        
        # モジュール内のクラスを探す
        for attr_name in dir(addon_module):
            attr = getattr(addon_module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'register_addon'):
                try:
                    addon = attr.register_addon(node)
                    addons.append(addon)
                    node.get_logger().info(f'アドオン {module_name}.{attr_name} をロードしました。')
                except Exception as e:
                    node.get_logger().error(f'アドオン {module_name}.{attr_name} のロード中にエラーが発生しました: {str(e)}')
                break
        else:
            node.get_logger().warn(f'アドオンモジュール {module_name} に @addon デコレータを使用したクラスが見つかりません。')
    return addons

def main(args=None):
    """
    メイン関数。
    DINDノードを初期化し、必要なコンポーネントを設定して実行します。
    """
    args = parse()
    rclpy.init()
    dind = DIND()
    load_config(dind)
    
    dind.visualize = args.visualize
    dind.exp_num = args.exp
    
    if args.camera == 'rs':
        camera = RealsenseCamera()
    elif args.camera == 'ds':
        camera = DatasetCamera(dind, args.ds_dir, start_time=args.time)
    elif args.camera == 'face':
        camera = FaceCamera()
    else:
        camera = RealsenseCamera()
        
    dind.set_camera(camera)
        
    coords_converter = camera.get_coords_converter()
    dind.set_coords_converter(coords_converter)
    
    exclude_list = []
    include_list = []
    
    if args.sync == 'yolo':
        observer = ObserverYolo(dind)
        dind.set_observer(observer)
    else:
        dind.set_observer(Observer3D(dind))
    if args.camera == 'face':
        dind.set_observer(ObserverFace(dind))
        
    if args.sync == 'ray':
        dind.set_data_sync(ImprovedRaySync(dind))
    elif args.sync == 'face':
        dind.set_data_sync(FaceSync(dind))
    elif args.sync == 'yolo':
        dind.set_data_sync(None)
        exclude_list.append('result_csv.py')
        exclude_list.append('result_show.py')
    elif args.camera == 'face':
        dind.set_data_sync(FaceSync(dind))
        exclude_list.append('result_csv.py')
        exclude_list.append('result_show.py')
        exclude_list.append('show_name.py')
        
    if not args.output:
        exclude_list.append('result_csv.py')
    if not args.visualize:
        exclude_list.append('result_show.py')
        
    app = dind.config_dict.get(f'config.apps.{args.app}', None)
    if app is None:
        dind.get_logger().error(f'アプリケーション {args.app} が不明のためdefaultで起動します。')
        app = dind.config_dict.get('config.apps.default', None)
        if app is None:
            dind.get_logger().error('defaultのアプリケーションが不明のため起動しません。')
            return
    else:
        #appはファイル名を格納したリスト
        include_list.extend(app)
        
    addons = load_addons(dind, include=include_list, exclude=exclude_list)
    executor = MultiThreadedExecutor()
    executor.add_node(dind)
    executor.spin()
    dind.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

def parse():
    clean_args = rclpy.utilities.remove_ros_args(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='rs', help='カメラの種類 (rs, ds)')
    parser.add_argument('--ds_dir', type=str, default='dataset/20250228_1738', help='データセットディレクトリのパス')
    parser.add_argument('--sync', type=str, default='ray', help='DataSyncの種類 (ray)')
    parser.add_argument('--time', type=int, default=None, help='データセットの開始時間 (ミリ秒)')
    parser.add_argument('-v', '--visualize', action='store_true', help='結果を表示する')
    parser.add_argument('--app', type=str, default='default', help='アプリケーションの設定')
    parser.add_argument('--exp', type=int, default=0, help='実験番号')
    parser.add_argument('-o', '--output', action='store_true', help='結果をcsvに出力する')
    args = parser.parse_args(clean_args[1:])
    return args

def load_config(dind: DIND):
    config = YamlConfig('config', 'config')
    config.add_default('yolo_model', 'yolo11m-pose.engine')
    config.add_default('frame.height', 480)
    config.add_default('frame.width', 640)

    config.add_default('apps.default', [
        'new_bring_candy.py'
        'result_csv.py'
        'result_show.py'
    ])
    
    dind.frame_height = config.get('frame.height', 480)
    dind.frame_width = config.get('frame.width', 640)
    
    dind.config_dict.update(config.get_dict())