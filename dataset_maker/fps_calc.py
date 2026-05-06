#!/usr/bin/env python3
"""
CSVファイルのタイムスタンプから平均FPSを計算するプログラム

タイムスタンプはミリ秒単位を想定。

使用方法:
    python calc_fps.py <csv_file>
"""

import argparse
import csv
from pathlib import Path


def calc_fps(csv_path: str) -> dict:
    """
    CSVファイルからFPS統計を計算
    
    Args:
        csv_path: CSVファイルパス
    
    Returns:
        FPS統計の辞書
    """
    # タイムスタンプを読み込み（ユニークなもののみ）
    timestamps: set[int] = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.add(int(row['timestamp']))
    
    if len(timestamps) < 2:
        raise ValueError("FPS計算には2フレーム以上必要です")
    
    # ソートしてリスト化
    sorted_timestamps = sorted(timestamps)
    
    # フレーム間隔を計算
    intervals = []
    for i in range(1, len(sorted_timestamps)):
        interval_ms = sorted_timestamps[i] - sorted_timestamps[i - 1]
        intervals.append(interval_ms)
    
    # 統計計算
    total_duration_ms = sorted_timestamps[-1] - sorted_timestamps[0]
    num_frames = len(sorted_timestamps)
    num_intervals = len(intervals)
    
    avg_interval_ms = sum(intervals) / num_intervals
    min_interval_ms = min(intervals)
    max_interval_ms = max(intervals)
    
    # FPS計算
    avg_fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0
    max_fps = 1000.0 / min_interval_ms if min_interval_ms > 0 else 0
    min_fps = 1000.0 / max_interval_ms if max_interval_ms > 0 else 0
    
    # 全体のFPS（総フレーム数 / 総時間）
    overall_fps = (num_frames - 1) / (total_duration_ms / 1000.0) if total_duration_ms > 0 else 0
    
    return {
        'num_frames': num_frames,
        'total_duration_ms': total_duration_ms,
        'total_duration_sec': total_duration_ms / 1000.0,
        'avg_interval_ms': avg_interval_ms,
        'min_interval_ms': min_interval_ms,
        'max_interval_ms': max_interval_ms,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'overall_fps': overall_fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CSVのタイムスタンプから平均FPSを計算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="CSVファイルパス"
    )
    
    args = parser.parse_args()
    
    stats = calc_fps(args.csv_file)
    
    print(f"\n{'=' * 40}")
    print(f" FPS Statistics: {Path(args.csv_file).name}")
    print(f"{'=' * 40}")
    print(f"  Frames:        {stats['num_frames']:>10d}")
    print(f"  Duration:      {stats['total_duration_sec']:>10.2f} sec")
    print(f"{'=' * 40}")
    print(f"  Average FPS:   {stats['avg_fps']:>10.2f}")
    print(f"  Min FPS:       {stats['min_fps']:>10.2f}")
    print(f"  Max FPS:       {stats['max_fps']:>10.2f}")
    print(f"  Overall FPS:   {stats['overall_fps']:>10.2f}")
    print(f"{'=' * 40}")
    print(f"  Avg interval:  {stats['avg_interval_ms']:>10.2f} ms")
    print(f"  Min interval:  {stats['min_interval_ms']:>10.2f} ms")
    print(f"  Max interval:  {stats['max_interval_ms']:>10.2f} ms")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()