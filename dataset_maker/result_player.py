#!/usr/bin/env python3
"""
トラッキング結果再生・動画出力プログラム

CSVファイルと画像フォルダを参照し、トラッキング結果を再生または動画出力する。
タイムスタンプを参照してFPSを再現。

使用方法:
    # 再生モード
    python playback_tracking.py <image_dir> <csv_file>
    
    # 動画出力モード
    python playback_tracking.py <image_dir> <csv_file> -o output.mp4
    
    # 再生速度変更
    python playback_tracking.py <image_dir> <csv_file> --speed 2.0
"""

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# 情報パネルの高さ
INFO_PANEL_HEIGHT = 120


@dataclass
class BoundingBox:
    """バウンディングボックスデータ"""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass 
class FrameData:
    """1フレームのデータ"""
    timestamp: int  # ミリ秒
    image_path: Path
    boxes: list[BoundingBox]


def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """トラックIDに基づいて一貫した色を返す"""
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


def load_tracking_data(csv_path: str, image_dir: Path) -> list[FrameData]:
    """
    CSVと画像ディレクトリからフレームデータを読み込む
    CSVに含まれるタイムスタンプのフレームのみを読み込む
    """
    # CSVからデータを読み込み
    data_by_timestamp: dict[str, list[BoundingBox]] = defaultdict(list)
    timestamps_in_csv: set[str] = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row['timestamp']
            timestamps_in_csv.add(timestamp)
            box = BoundingBox(
                track_id=int(row['id']),
                x1=float(row['x1']),
                y1=float(row['y1']),
                x2=float(row['x2']),
                y2=float(row['y2'])
            )
            data_by_timestamp[timestamp].append(box)
    
    # CSVに含まれるタイムスタンプの画像のみを読み込む
    frames: list[FrameData] = []
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for timestamp_str in timestamps_in_csv:
        # 対応する画像ファイルを探す
        img_path = None
        for ext in extensions:
            candidate = image_dir / f"{timestamp_str}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"Warning: Image not found for timestamp {timestamp_str}")
            continue
        
        try:
            timestamp_ms = int(timestamp_str)
        except ValueError:
            print(f"Warning: Invalid timestamp format: {timestamp_str}")
            continue
        
        boxes = data_by_timestamp.get(timestamp_str, [])
        frames.append(FrameData(
            timestamp=timestamp_ms,
            image_path=img_path,
            boxes=boxes
        ))
    
    # タイムスタンプでソート
    frames.sort(key=lambda f: f.timestamp)
    
    return frames


def draw_frame(
    image: np.ndarray,
    frame_data: FrameData,
    frame_idx: int,
    total_frames: int,
    current_fps: float,
    elapsed_time_sec: float
) -> np.ndarray:
    """フレームを描画（情報パネルは画像の下に配置）"""
    h, w = image.shape[:2]
    
    # 画像の下に情報パネル用のスペースを追加
    img = np.zeros((h + INFO_PANEL_HEIGHT, w, 3), dtype=np.uint8)
    img[:h, :] = image
    
    # バウンディングボックスを描画
    for box in frame_data.boxes:
        color = get_color_for_id(box.track_id)
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        
        # バウンディングボックス
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # IDラベル
        label = f"ID:{box.track_id}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 5 if y1 > label_h + 10 else y2 + label_h + 10
        cv2.rectangle(img, (x1, label_y - label_h - 5), (x1 + label_w + 10, label_y + 5), color, -1)
        cv2.putText(img, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 情報パネルの背景（下部）
    panel_y = h
    cv2.rectangle(img, (0, panel_y), (w, panel_y + INFO_PANEL_HEIGHT), (30, 30, 30), -1)
    
    # 区切り線
    cv2.line(img, (0, panel_y), (w, panel_y), (80, 80, 80), 2)
    
    # 左側: フレーム情報
    y_offset = panel_y + 25
    cv2.putText(img, f"Frame: {frame_idx + 1}/{total_frames}", 
                (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img, f"Timestamp: {frame_data.timestamp} ms",
                (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    y_offset += 25
    cv2.putText(img, f"Detections: {len(frame_data.boxes)}",
                (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    y_offset += 25
    ids = sorted([box.track_id for box in frame_data.boxes])
    ids_str = ", ".join(map(str, ids)) if ids else "None"
    cv2.putText(img, f"IDs: {ids_str}",
                (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 右側: 再生情報
    right_x = w - 250
    y_offset = panel_y + 25
    cv2.putText(img, f"Elapsed: {elapsed_time_sec:.2f} sec",
                (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img, f"Current FPS: {current_fps:.1f}",
                (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 中央: 操作説明（再生モード時のみ意味がある）
    center_x = w // 2 - 100
    y_offset = panel_y + INFO_PANEL_HEIGHT - 15
    cv2.putText(img, "[Space] Pause  [Q] Quit  [<][>] Step",
                (center_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    
    return img


def playback(
    frames: list[FrameData],
    speed: float = 1.0,
    display_scale: float = 1.0
) -> None:
    """
    トラッキング結果を再生
    各フレーム間のタイムスタンプ間隔を正確に再現
    """
    if not frames:
        print("No frames to play")
        return
    
    cv2.namedWindow("Tracking Playback", cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    paused = False
    base_timestamp = frames[0].timestamp
    
    print("\n=== Playback Controls ===")
    print("  [Space] : Pause/Resume")
    print("  [Q]     : Quit")
    print("  [←][→] : Step frame (when paused)")
    print("  [R]     : Restart")
    print("=========================\n")
    
    while True:
        frame_data = frames[frame_idx]
        
        # 画像読み込み
        img = cv2.imread(str(frame_data.image_path))
        if img is None:
            print(f"Warning: Could not load {frame_data.image_path}")
            frame_idx = (frame_idx + 1) % len(frames)
            continue
        
        # 経過時間とFPS計算
        if frame_idx > 0:
            interval_ms = frame_data.timestamp - frames[frame_idx - 1].timestamp
            current_fps = 1000.0 / interval_ms if interval_ms > 0 else 0
        else:
            current_fps = 0
        
        elapsed_time_sec = (frame_data.timestamp - base_timestamp) / 1000.0
        
        # 描画
        vis_img = draw_frame(img, frame_data, frame_idx, len(frames), current_fps, elapsed_time_sec)
        
        # スケール調整
        if display_scale != 1.0:
            new_w = int(vis_img.shape[1] * display_scale)
            new_h = int(vis_img.shape[0] * display_scale)
            vis_img = cv2.resize(vis_img, (new_w, new_h))
        
        cv2.imshow("Tracking Playback", vis_img)
        
        # 次フレームまでの待機時間を計算（フレーム間隔を正確に再現）
        if not paused and frame_idx < len(frames) - 1:
            # 次フレームとの時間差をそのまま使用
            next_timestamp = frames[frame_idx + 1].timestamp
            interval_ms = next_timestamp - frame_data.timestamp
            # speedで調整し、最低1msは待機
            wait_ms = max(1, int(interval_ms / speed))
        else:
            wait_ms = 0 if paused else 1
        
        # このフレームの表示開始時刻を記録
        frame_start_time = time.time()
        
        # キー入力（待機時間の一部として処理）
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            frame_idx = 0
            paused = False
            continue
        elif paused:
            if key == 81 or key == 2:  # Left arrow
                frame_idx = max(0, frame_idx - 1)
            elif key == 83 or key == 3:  # Right arrow
                frame_idx = min(len(frames) - 1, frame_idx + 1)
            continue  # pausedの場合はここでループ継続
        
        # 残りの待機時間をスリープ（正確なタイミング再現のため）
        if not paused and frame_idx < len(frames) - 1:
            elapsed_ms = (time.time() - frame_start_time) * 1000
            remaining_ms = wait_ms - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000.0)
        
        # 次フレームへ
        if not paused:
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            else:
                # ループ再生
                frame_idx = 0
    
    cv2.destroyAllWindows()


def export_video(
    frames: list[FrameData],
    output_path: str,
    output_fps: float = 30.0,
    display_scale: float = 1.0
) -> None:
    """
    トラッキング結果を動画ファイルに出力
    フレーム間隔を再現するため、出力FPSに合わせてフレームを複製/スキップ
    """
    if not frames:
        print("No frames to export")
        return
    
    print(f"Exporting video at {output_fps:.2f} FPS (with timing preservation)...")
    
    # 最初のフレームでサイズを決定
    first_img = cv2.imread(str(frames[0].image_path))
    if first_img is None:
        raise ValueError(f"Could not load first frame: {frames[0].image_path}")
    
    h, w = first_img.shape[:2]
    output_h = int((h + INFO_PANEL_HEIGHT) * display_scale)
    output_w = int(w * display_scale)
    
    # VideoWriter設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_w, output_h))
    
    if not out.isOpened():
        raise ValueError(f"Could not open video writer: {output_path}")
    
    base_timestamp = frames[0].timestamp
    frame_duration_ms = 1000.0 / output_fps  # 出力動画の1フレームあたりの時間
    
    total_written = 0
    
    for i, frame_data in enumerate(frames):
        img = cv2.imread(str(frame_data.image_path))
        if img is None:
            print(f"Warning: Could not load {frame_data.image_path}, skipping")
            continue
        
        # FPS計算
        if i > 0:
            interval_ms = frame_data.timestamp - frames[i - 1].timestamp
            current_fps = 1000.0 / interval_ms if interval_ms > 0 else 0
        else:
            interval_ms = frame_duration_ms
            current_fps = output_fps
        
        elapsed_time_sec = (frame_data.timestamp - base_timestamp) / 1000.0
        
        # 描画
        vis_img = draw_frame(img, frame_data, i, len(frames), current_fps, elapsed_time_sec)
        
        # スケール調整
        if display_scale != 1.0:
            vis_img = cv2.resize(vis_img, (output_w, output_h))
        
        # フレーム間隔を再現するために必要なフレーム数を計算
        # （このフレームを何回書き込むか）
        num_repeats = max(1, round(interval_ms / frame_duration_ms))
        
        for _ in range(num_repeats):
            out.write(vis_img)
            total_written += 1
        
        # 進捗表示
        if (i + 1) % 100 == 0 or (i + 1) == len(frames):
            print(f"  Processed {i + 1}/{len(frames)} source frames ({total_written} output frames)")
    
    out.release()
    
    actual_duration = total_written / output_fps
    original_duration = (frames[-1].timestamp - frames[0].timestamp) / 1000.0
    
    print(f"\nVideo saved: {output_path}")
    print(f"  - Resolution: {output_w}x{output_h}")
    print(f"  - Output FPS: {output_fps:.2f}")
    print(f"  - Total frames written: {total_written}")
    print(f"  - Video duration: {actual_duration:.2f} sec")
    print(f"  - Original duration: {original_duration:.2f} sec")


def main():
    parser = argparse.ArgumentParser(
        description="トラッキング結果の再生・動画出力",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="画像ディレクトリパス"
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
        help="動画出力パス（指定しない場合は再生モード）"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="出力動画のFPS（フレーム間隔は複製で再現）"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="再生速度（再生モード時のみ）"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="表示/出力スケール"
    )
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    print(f"Loading data...")
    frames = load_tracking_data(args.csv_file, image_dir)
    print(f"  - Loaded {len(frames)} frames")
    
    if not frames:
        print("Error: No frames found")
        return
    
    if args.output:
        # 動画出力モード
        export_video(frames, args.output, args.fps, args.scale)
    else:
        # 再生モード
        playback(frames, args.speed, args.scale)


if __name__ == "__main__":
    main()