# iSpace-v4


## 対応環境・言語

([kachaka-api](https://github.com/pf-robotics/kachaka-api/)に準拠)
- Ubuntu 22.04 LTS
- Python 3.10
- ROS2 Humble

## 導入方法

### ROS 2 Humbleのセットアップ

- 以下を参考に、ROS 2 Humbleをセットアップしてください。
  - https://docs.ros.org/en/humble/index.html
  - 大体ja_JP.UTF-8になっていると思うので、Set localeは不要です。Setup Sourcesから始めてください。
  - Desktop InstallとDevelopment toolsのみインストールしてください。ROS-Baseは不要です。

### iSpace-v4のダウンロード

```
cd ~
git clone https://github.com/minesaki-ais/iSpace-v4.git
```

### ワークスペースの作成

- 以下の手順でワークスペースを作成してください。

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s ~/iSpace-v4/ispace_dind/ ispace_dind
ln -s ~/iSpace-v4/ispace_interfaces/ ispace_interfaces
```

### ビルド方法

```
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build
```

## 起動方法

### 起動コマンド
ros2_ws/src 内で以下のコマンドを実行。
```
ros2 run ispace_dind dind <option>
```

### オプション一覧

| option | 型 | default値 | 説明 |
| --- | --- | --- | --- |
| --camera | str | rs | カメラの種類を指定（rs=realsense/ds=dataset） |
| --ds_dir | str | dataset/20250228_1738 | 使用するデータセットのディレクトリを指定<br>（データセットカメラ指定時のみ有効）
| -v | bool | False | 画面の表示/非表示 |
| -o | bool | False | 結果のCSV保存 有効/無効 |
| --time | int | None | データセット開始時刻の設定[ms]<br>負の値の場合、指定ms後に動作開始<br>Noneの場合他のDINDと自動同期


