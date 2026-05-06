#!/usr/bin/env python3
"""
トラッキング結果修正GUIアプリケーション

CSVファイルに保存されたトラッキング結果を修正するためのツール。
- バウンディングボックスの削除（Backspace）
- IDスイッチングの修正（1-9キー）
- フレーム移動（左右キー）

使用方法:
    python tracking_editor.py <image_dir> <csv_file> [-o output.csv]
"""

import argparse
import csv
import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# 情報パネルの高さ
INFO_PANEL_HEIGHT = 90


@dataclass
class BoundingBox:
    """バウンディングボックスデータ"""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    
    def contains(self, x: int, y: int) -> bool:
        """指定座標がボックス内にあるか判定"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def area(self) -> float:
        """ボックスの面積を計算"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class FrameData:
    """1フレームのデータ"""
    timestamp: str
    boxes: list[BoundingBox] = field(default_factory=list)


@dataclass
class EditorState:
    """エディタの状態"""
    frames: list[FrameData]
    current_frame_idx: int = 0
    selected_box_idx: int | None = None
    modified: bool = False
    
    @property
    def current_frame(self) -> FrameData:
        return self.frames[self.current_frame_idx]
    
    @property
    def total_frames(self) -> int:
        return len(self.frames)


def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """トラックIDに基づいて一貫した色を返す"""
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


def load_tracking_data(csv_path: str, image_dir: Path) -> list[FrameData]:
    """
    CSVファイルからトラッキングデータを読み込む
    画像ディレクトリに存在するすべてのフレームを含める（検出がないフレームも）
    """
    # 画像ファイル一覧を取得
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in extensions],
        key=lambda p: p.stem
    )
    
    # タイムスタンプ -> FrameData のマッピングを作成
    frames_dict: dict[str, FrameData] = {}
    for img_file in image_files:
        timestamp = img_file.stem
        frames_dict[timestamp] = FrameData(timestamp=timestamp, boxes=[])
    
    # CSVからデータを読み込み
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row['timestamp']
            if timestamp in frames_dict:
                box = BoundingBox(
                    track_id=int(row['id']),
                    x1=float(row['x1']),
                    y1=float(row['y1']),
                    x2=float(row['x2']),
                    y2=float(row['y2'])
                )
                frames_dict[timestamp].boxes.append(box)
    
    # タイムスタンプ順にソートしてリストに変換
    frames = [frames_dict[img.stem] for img in image_files if img.stem in frames_dict]
    
    return frames


def save_tracking_data(frames: list[FrameData], output_path: str) -> None:
    """トラッキングデータをCSVファイルに保存"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'id', 'x1', 'y1', 'x2', 'y2'])
        
        for frame in frames:
            for box in frame.boxes:
                writer.writerow([
                    frame.timestamp,
                    box.track_id,
                    round(box.x1, 2),
                    round(box.y1, 2),
                    round(box.x2, 2),
                    round(box.y2, 2)
                ])


def draw_frame(
    image: np.ndarray,
    frame_data: FrameData,
    selected_idx: int | None,
    frame_idx: int,
    total_frames: int,
    modified: bool
) -> np.ndarray:
    """フレームを描画（情報パネルは画像の下に配置）"""
    h, w = image.shape[:2]
    
    # 画像の下に情報パネル用のスペースを追加
    img = np.zeros((h + INFO_PANEL_HEIGHT, w, 3), dtype=np.uint8)
    img[:h, :] = image  # 元画像を上部にコピー
    
    # 情報パネルの背景（下部）
    cv2.rectangle(img, (0, h), (w, h + INFO_PANEL_HEIGHT), (40, 40, 40), -1)
    
    # フレーム情報
    status = "[Modified]" if modified else ""
    cv2.putText(
        img, f"Frame: {frame_idx + 1}/{total_frames} | Timestamp: {frame_data.timestamp} {status}",
        (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        img, f"Detections: {len(frame_data.boxes)}",
        (10, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
    )
    cv2.putText(
        img, "Keys: [<-][->] Move | [1-9] Change ID | [Backspace] Delete | [S] Save | [Q] Quit",
        (10, h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
    )
    
    # バウンディングボックスを描画
    for i, box in enumerate(frame_data.boxes):
        color = get_color_for_id(box.track_id)
        thickness = 3 if i == selected_idx else 2
        
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        
        # 選択中のボックスはハイライト
        if i == selected_idx:
            # 外枠を白で描画
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
        
        # バウンディングボックス
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # IDラベルの背景
        label = f"ID: {box.track_id}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # ラベルはボックスの上に配置（画面外にはみ出す場合は内側に）
        label_y = y1 - 5 if y1 > label_h + 15 else y2 + label_h + 10
        cv2.rectangle(img, (x1, label_y - label_h - 5), (x1 + label_w + 10, label_y + 5), color, -1)
        cv2.putText(img, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img


class TrackingEditor:
    """トラッキング結果エディタ"""
    
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        output_path: str,
        display_scale: float = 1.0
    ):
        self.image_dir = Path(image_dir)
        self.csv_path = csv_path
        self.output_path = output_path
        self.display_scale = display_scale
        
        # 出力ファイルが存在する場合はそちらを読み込む（途中保存からの再開）
        load_path = csv_path
        if output_path != csv_path and Path(output_path).exists():
            print(f"Found existing output file: {output_path}")
            print("Loading from output file to resume editing...")
            load_path = output_path
        
        # データ読み込み
        print(f"Loading tracking data from: {load_path}")
        frames = load_tracking_data(load_path, self.image_dir)
        if not frames:
            raise ValueError("No frames found!")
        
        self.state = EditorState(frames=frames)
        
        # 元の状態を保存（リセット用に元のCSVから読み込む）
        self.original_state = load_tracking_data(csv_path, self.image_dir)
        
        print(f"Loaded {len(frames)} frames")
        
        # ウィンドウ設定
        self.window_name = "Tracking Editor"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _get_image_path(self, timestamp: str) -> Path | None:
        """タイムスタンプに対応する画像パスを取得"""
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = self.image_dir / f"{timestamp}{ext}"
            if path.exists():
                return path
        return None
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """マウスイベントコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # スケールを考慮して座標を変換
            real_x = int(x / self.display_scale)
            real_y = int(y / self.display_scale)
            
            # クリック位置にあるボックスを探す（面積が小さい順にソート）
            candidates: list[tuple[int, float]] = []  # (index, area)
            for i, box in enumerate(self.state.current_frame.boxes):
                if box.contains(real_x, real_y):
                    candidates.append((i, box.area()))
            
            # 面積が最小のボックスを選択（重なっている場合は小さい方が選択される）
            if candidates:
                candidates.sort(key=lambda x: x[1])  # 面積でソート
                self.state.selected_box_idx = candidates[0][0]
                box = self.state.current_frame.boxes[self.state.selected_box_idx]
                print(f"Selected box ID: {box.track_id}")
            else:
                self.state.selected_box_idx = None
            
            self._update_display()
    
    def _load_current_image(self) -> np.ndarray | None:
        """現在のフレームの画像を読み込む"""
        timestamp = self.state.current_frame.timestamp
        img_path = self._get_image_path(timestamp)
        
        if img_path is None:
            print(f"Warning: Image not found for timestamp: {timestamp}")
            return None
        
        return cv2.imread(str(img_path))
    
    def _update_display(self) -> None:
        """表示を更新"""
        img = self._load_current_image()
        if img is None:
            # 画像がない場合は黒画像を表示
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        vis_img = draw_frame(
            img,
            self.state.current_frame,
            self.state.selected_box_idx,
            self.state.current_frame_idx,
            self.state.total_frames,
            self.state.modified
        )
        
        # スケール調整
        if self.display_scale != 1.0:
            new_w = int(vis_img.shape[1] * self.display_scale)
            new_h = int(vis_img.shape[0] * self.display_scale)
            vis_img = cv2.resize(vis_img, (new_w, new_h))
        
        cv2.imshow(self.window_name, vis_img)
    
    def _delete_selected_box(self) -> None:
        """選択中のボックスを削除"""
        if self.state.selected_box_idx is not None:
            box = self.state.current_frame.boxes[self.state.selected_box_idx]
            print(f"Deleted box ID: {box.track_id}")
            del self.state.current_frame.boxes[self.state.selected_box_idx]
            self.state.selected_box_idx = None
            self.state.modified = True
            self._update_display()
    
    def _change_id_from_current_frame(self, new_id: int) -> None:
        """
        選択中のボックスのIDを現在のフレーム以降すべて変更する
        IDスイッチングの修正用
        """
        if self.state.selected_box_idx is None:
            print("No box selected")
            return
        
        old_id = self.state.current_frame.boxes[self.state.selected_box_idx].track_id
        
        if old_id == new_id:
            print(f"ID is already {new_id}")
            return
        
        # 現在のフレーム以降で、old_id と new_id を持つボックスを交換
        change_count = 0
        for frame in self.state.frames[self.state.current_frame_idx:]:
            for box in frame.boxes:
                if box.track_id == old_id:
                    box.track_id = new_id
                    change_count += 1
                elif box.track_id == new_id:
                    box.track_id = old_id
                    change_count += 1
        
        print(f"Swapped ID {old_id} <-> {new_id} in {change_count} boxes from frame {self.state.current_frame_idx + 1}")
        self.state.modified = True
        self._update_display()
    
    def _move_frame(self, delta: int) -> None:
        """フレームを移動"""
        new_idx = self.state.current_frame_idx + delta
        new_idx = max(0, min(new_idx, self.state.total_frames - 1))
        
        if new_idx != self.state.current_frame_idx:
            self.state.current_frame_idx = new_idx
            self.state.selected_box_idx = None
            self._update_display()
    
    def _save(self) -> None:
        """現在の状態を保存"""
        save_tracking_data(self.state.frames, self.output_path)
        print(f"Saved to: {self.output_path}")
        self.state.modified = False
        self._update_display()
    
    def _confirm_quit(self) -> bool:
        """終了確認"""
        if not self.state.modified:
            return True
        
        print("\nUnsaved changes! Press 'q' again to quit without saving, or 's' to save.")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            return True
        elif key == ord('s'):
            self._save()
            return True
        return False
    
    def run(self) -> None:
        """エディタのメインループ"""
        print("\n=== Tracking Editor ===")
        print("Controls:")
        print("  [←][→]     : Move between frames")
        print("  [Click]    : Select bounding box")
        print("  [1-9]      : Change selected box ID (swaps from current frame onward)")
        print("  [Backspace]: Delete selected box")
        print("  [S]        : Save changes")
        print("  [R]        : Reset to original")
        print("  [Q]        : Quit")
        print("========================\n")
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # 終了
            if key == ord('q'):
                if self._confirm_quit():
                    break
            
            # 保存
            elif key == ord('s'):
                self._save()
            
            # リセット
            elif key == ord('r'):
                self.state.frames = copy.deepcopy(self.original_state)
                self.state.modified = False
                self.state.selected_box_idx = None
                print("Reset to original state")
                self._update_display()
            
            # フレーム移動（左矢印）
            elif key == 81 or key == 2:  # Left arrow (Linux / Windows)
                self._move_frame(-1)
            
            # フレーム移動（右矢印）
            elif key == 83 or key == 3:  # Right arrow (Linux / Windows)
                self._move_frame(1)
            
            # Page Up / Page Down で10フレームジャンプ
            elif key == 85:  # Page Up
                self._move_frame(-10)
            elif key == 86:  # Page Down
                self._move_frame(10)
            
            # Home / End でフレームの最初/最後へ
            elif key == 80:  # Home
                self._move_frame(-self.state.total_frames)
            elif key == 87:  # End
                self._move_frame(self.state.total_frames)
            
            # バックスペースで削除
            elif key == 8 or key == 127:  # Backspace
                self._delete_selected_box()
            
            # 1-9キーでID変更
            elif ord('1') <= key <= ord('9'):
                new_id = key - ord('0')
                self._change_id_from_current_frame(new_id)
        
        cv2.destroyAllWindows()
        print("Editor closed.")


def main():
    parser = argparse.ArgumentParser(
        description="トラッキング結果修正エディタ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="画像が格納されているディレクトリパス"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="トラッキング結果CSVファイルパス"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力CSVファイルパス（指定しない場合は入力ファイルを上書き）"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="表示画像のスケール（1.0=原寸）"
    )
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else args.csv_file
    
    editor = TrackingEditor(
        image_dir=args.image_dir,
        csv_path=args.csv_file,
        output_path=output_path,
        display_scale=args.scale
    )
    editor.run()


if __name__ == "__main__":
    main()