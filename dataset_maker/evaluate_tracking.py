#!/usr/bin/env python3
"""
トラッキング評価プログラム

Ground Truth CSVと予測結果CSVを比較し、
HOTA, MOTA, IDF1などのトラッキング精度を評価する。

TrackEvalライブラリを使用。

使用方法:
    python evaluate_tracking.py <gt_csv> <pred_csv> [options]

出力メトリクス:
    - HOTA: Higher Order Tracking Accuracy（検出と関連付けのバランス）
    - DetA: Detection Accuracy
    - AssA: Association Accuracy
    - MOTA: Multiple Object Tracking Accuracy
    - MOTP: Multiple Object Tracking Precision
    - IDF1: ID F1 Score
    - その他多数
"""

import argparse
import csv
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import trackeval
except ImportError:
    print("TrackEvalがインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("  pip install trackeval")
    exit(1)


def load_csv_data(csv_path: str) -> dict[str, list[dict]]:
    """
    CSVファイルを読み込み、タイムスタンプごとにグループ化
    
    Returns:
        {timestamp: [{'id': int, 'x1': float, 'y1': float, 'x2': float, 'y2': float}, ...]}
    """
    data = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = row['timestamp']
            data[timestamp].append({
                'id': int(row['id']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2'])
            })
    
    return dict(data)


def convert_to_mot_format(
    data: dict[str, list[dict]],
    output_path: Path,
    timestamps: list[str]
) -> None:
    """
    独自CSV形式をMOTChallenge形式に変換
    
    MOTChallenge形式:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    
    Args:
        data: タイムスタンプごとのデータ
        output_path: 出力ファイルパス
        timestamps: タイムスタンプのリスト（フレーム番号の順序を決定）
    """
    # タイムスタンプ -> フレーム番号のマッピング
    timestamp_to_frame = {ts: i + 1 for i, ts in enumerate(timestamps)}
    
    with open(output_path, 'w') as f:
        for timestamp, detections in data.items():
            if timestamp not in timestamp_to_frame:
                continue
            
            frame_id = timestamp_to_frame[timestamp]
            
            for det in detections:
                # x1, y1, x2, y2 -> left, top, width, height
                left = det['x1']
                top = det['y1']
                width = det['x2'] - det['x1']
                height = det['y2'] - det['y1']
                
                # MOTChallenge形式: frame, id, left, top, width, height, conf, x, y, z
                # conf=1（確信度）, x,y,z=-1（3D座標は使用しない）
                f.write(f"{frame_id},{det['id']},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")


def create_seqinfo(output_path: Path, seq_length: int, name: str = "eval_seq") -> None:
    """seqinfo.iniファイルを作成"""
    content = f"""[Sequence]
name={name}
imDir=img1
frameRate=30
seqLength={seq_length}
imWidth=1920
imHeight=1080
imExt=.jpg
"""
    with open(output_path, 'w') as f:
        f.write(content)


def setup_trackeval_structure(
    gt_data: dict[str, list[dict]],
    pred_data: dict[str, list[dict]],
    work_dir: Path
) -> tuple[Path, Path, list[str], int]:
    """
    TrackEval用のディレクトリ構造を作成
    
    構造:
        work_dir/
            gt/
                mot_challenge/
                    EVAL-train/
                        eval_seq/
                            gt/
                                gt.txt
                            seqinfo.ini
                seqmaps/
                    EVAL-train.txt
            trackers/
                mot_challenge/
                    EVAL-train/
                        tracker/
                            data/
                                eval_seq.txt
    """
    # predictionsに含まれるタイムスタンプのみを評価対象とする
    # （フレーム落ちを考慮：GTにあってもpredictionsにないフレームは評価から除外）
    all_timestamps = sorted(set(gt_data.keys()) & set(pred_data.keys()))
    seq_length = len(all_timestamps)
    
    if seq_length == 0:
        raise ValueError("No common timestamps found between GT and predictions")
    
    # GTのみ、Predのみのフレーム数を報告
    gt_only = set(gt_data.keys()) - set(pred_data.keys())
    pred_only = set(pred_data.keys()) - set(gt_data.keys())
    if gt_only:
        print(f"  - Note: {len(gt_only)} GT frames excluded (not in predictions)")
    if pred_only:
        print(f"  - Note: {len(pred_only)} prediction frames excluded (not in GT)")
    
    # ベンチマーク名とシーケンス名
    benchmark = "EVAL"
    split = "train"
    seq_name = "eval_seq"
    benchmark_split = f"{benchmark}-{split}"
    
    # GT用ディレクトリ作成
    gt_base = work_dir / "gt" / "mot_challenge"
    gt_seq_dir = gt_base / benchmark_split / seq_name
    gt_data_dir = gt_seq_dir / "gt"
    gt_data_dir.mkdir(parents=True, exist_ok=True)
    
    # seqmapsディレクトリ作成
    seqmaps_dir = gt_base / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    
    # Tracker用ディレクトリ作成
    tracker_base = work_dir / "trackers" / "mot_challenge"
    tracker_data_dir = tracker_base / benchmark_split / "tracker" / "data"
    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    
    # GTデータをMOT形式で保存
    convert_to_mot_format(gt_data, gt_data_dir / "gt.txt", all_timestamps)
    
    # seqinfo.ini作成（シーケンスディレクトリ直下）
    create_seqinfo(gt_seq_dir / "seqinfo.ini", seq_length, seq_name)
    
    # seqmapファイル作成
    with open(seqmaps_dir / f"{benchmark_split}.txt", 'w') as f:
        f.write("name\n")
        f.write(f"{seq_name}\n")
    
    # 予測データをMOT形式で保存
    convert_to_mot_format(pred_data, tracker_data_dir / f"{seq_name}.txt", all_timestamps)
    
    return (
        gt_base,
        tracker_base,
        all_timestamps,
        seq_length
    )


def run_evaluation(
    gt_csv: str,
    pred_csv: str,
    iou_threshold: float = 0.5,
    print_details: bool = True
) -> dict:
    """
    トラッキング評価を実行
    
    Args:
        gt_csv: Ground Truth CSVファイルパス
        pred_csv: 予測結果CSVファイルパス
        iou_threshold: マッチング用IoU閾値
        print_details: 詳細結果を表示するか
    
    Returns:
        評価結果の辞書
    """
    print(f"Loading Ground Truth: {gt_csv}")
    gt_data = load_csv_data(gt_csv)
    print(f"  - {len(gt_data)} frames, {sum(len(v) for v in gt_data.values())} detections")
    
    print(f"Loading Predictions: {pred_csv}")
    pred_data = load_csv_data(pred_csv)
    print(f"  - {len(pred_data)} frames, {sum(len(v) for v in pred_data.values())} detections")
    
    # 一時ディレクトリにTrackEval用構造を作成
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        
        print("\nSetting up evaluation structure...")
        gt_folder, tracker_folder, timestamps, seq_length = setup_trackeval_structure(
            gt_data, pred_data, work_dir
        )
        print(f"  - Sequence length: {seq_length} frames")
        
        # デバッグ: ディレクトリ構造を確認
        # import subprocess
        # subprocess.run(['find', str(work_dir), '-type', 'f'], check=True)
        
        # TrackEval設定
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['DISPLAY_LESS_PROGRESS'] = True
        eval_config['PRINT_CONFIG'] = False
        eval_config['PRINT_RESULTS'] = False
        eval_config['TIME_PROGRESS'] = False
        eval_config['OUTPUT_SUMMARY'] = False
        eval_config['OUTPUT_DETAILED'] = False
        eval_config['PLOT_CURVES'] = False
        
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config['GT_FOLDER'] = str(gt_folder)
        dataset_config['TRACKERS_FOLDER'] = str(tracker_folder)
        dataset_config['BENCHMARK'] = 'EVAL'
        dataset_config['SPLIT_TO_EVAL'] = 'train'
        dataset_config['TRACKERS_TO_EVAL'] = ['tracker']
        dataset_config['SKIP_SPLIT_FOL'] = False
        dataset_config['PRINT_CONFIG'] = False
        dataset_config['DO_PREPROC'] = False  # 前処理をスキップ
        
        # メトリクス設定
        metrics_config = {
            'METRICS': ['HOTA', 'CLEAR', 'Identity'],
            'THRESHOLD': iou_threshold
        }
        
        # 評価実行
        print("\nRunning evaluation...")
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = [
            trackeval.metrics.HOTA(metrics_config),
            trackeval.metrics.CLEAR(metrics_config),
            trackeval.metrics.Identity(metrics_config)
        ]
        
        raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)
        
        # 結果を抽出
        results = {}
        
        if 'MotChallenge2DBox' in raw_results:
            tracker_results = raw_results['MotChallenge2DBox']['tracker']
            if 'COMBINED_SEQ' in tracker_results:
                seq_results = tracker_results['COMBINED_SEQ']
                if 'pedestrian' in seq_results:
                    ped_results = seq_results['pedestrian']
                    
                    # HOTAメトリクス
                    if 'HOTA' in ped_results:
                        hota = ped_results['HOTA']
                        results['HOTA'] = np.mean(hota['HOTA']) * 100
                        results['DetA'] = np.mean(hota['DetA']) * 100
                        results['AssA'] = np.mean(hota['AssA']) * 100
                        results['DetRe'] = np.mean(hota['DetRe']) * 100
                        results['DetPr'] = np.mean(hota['DetPr']) * 100
                        results['AssRe'] = np.mean(hota['AssRe']) * 100
                        results['AssPr'] = np.mean(hota['AssPr']) * 100
                        results['LocA'] = np.mean(hota['LocA']) * 100
                    
                    # CLEARメトリクス
                    if 'CLEAR' in ped_results:
                        clear = ped_results['CLEAR']
                        results['MOTA'] = clear['MOTA'] * 100
                        results['MOTP'] = clear['MOTP'] * 100
                        results['MT'] = clear['MT']  # Mostly Tracked
                        results['PT'] = clear['PT']  # Partially Tracked
                        results['ML'] = clear['ML']  # Mostly Lost
                        results['FP'] = clear['CLR_FP']  # False Positives
                        results['FN'] = clear['CLR_FN']  # False Negatives
                        results['IDs'] = clear['IDSW']  # ID Switches
                        results['Frag'] = clear['Frag']  # Fragmentations
                        results['Recall'] = clear['CLR_Re'] * 100
                        results['Precision'] = clear['CLR_Pr'] * 100
                    
                    # Identityメトリクス
                    if 'Identity' in ped_results:
                        identity = ped_results['Identity']
                        results['IDF1'] = identity['IDF1'] * 100
                        results['IDP'] = identity['IDP'] * 100
                        results['IDR'] = identity['IDR'] * 100
        
        return results


def print_results(results: dict) -> None:
    """評価結果を整形して表示"""
    print("\n" + "=" * 60)
    print(" TRACKING EVALUATION RESULTS")
    print("=" * 60)
    
    if not results:
        print("No results available.")
        return
    
    # HOTAメトリクス
    print("\n[HOTA Metrics]")
    print("-" * 40)
    if 'HOTA' in results:
        print(f"  HOTA:  {results['HOTA']:6.2f}%  (Overall tracking accuracy)")
        print(f"  DetA:  {results['DetA']:6.2f}%  (Detection accuracy)")
        print(f"  AssA:  {results['AssA']:6.2f}%  (Association accuracy)")
        print(f"  LocA:  {results['LocA']:6.2f}%  (Localization accuracy)")
    
    # Identityメトリクス
    print("\n[Identity Metrics]")
    print("-" * 40)
    if 'IDF1' in results:
        print(f"  IDF1:  {results['IDF1']:6.2f}%  (ID F1 score)")
        print(f"  IDP:   {results['IDP']:6.2f}%  (ID precision)")
        print(f"  IDR:   {results['IDR']:6.2f}%  (ID recall)")
    
    # CLEARメトリクス
    print("\n[CLEAR Metrics]")
    print("-" * 40)
    if 'MOTA' in results:
        print(f"  MOTA:  {results['MOTA']:6.2f}%  (Multi-object tracking accuracy)")
        print(f"  MOTP:  {results['MOTP']:6.2f}%  (Multi-object tracking precision)")
    
    # カウント
    print("\n[Counts]")
    print("-" * 40)
    if 'FP' in results:
        print(f"  FP:    {int(results['FP']):6d}    (False positives)")
        print(f"  FN:    {int(results['FN']):6d}    (False negatives)")
        print(f"  IDs:   {int(results['IDs']):6d}    (ID switches)")
        print(f"  Frag:  {int(results['Frag']):6d}    (Fragmentations)")
    
    # トラック品質
    print("\n[Track Quality]")
    print("-" * 40)
    if 'MT' in results:
        print(f"  MT:    {int(results['MT']):6d}    (Mostly tracked)")
        print(f"  PT:    {int(results['PT']):6d}    (Partially tracked)")
        print(f"  ML:    {int(results['ML']):6d}    (Mostly lost)")
    
    print("\n" + "=" * 60)


def save_results_csv(results: dict, output_path: str) -> None:
    """評価結果をCSVファイルに保存"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.4f}"])
            else:
                writer.writerow([key, value])
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="トラッキング評価プログラム（HOTA, MOTA, IDF1等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
評価メトリクスの説明:
  HOTA  - Higher Order Tracking Accuracy（検出と関連付けのバランスを取った総合指標）
  DetA  - Detection Accuracy（検出精度）
  AssA  - Association Accuracy（関連付け精度）
  MOTA  - Multiple Object Tracking Accuracy（従来の総合指標、検出重視）
  IDF1  - ID F1 Score（ID一貫性の指標）

使用例:
  python evaluate_tracking.py ground_truth.csv predictions.csv
  python evaluate_tracking.py gt.csv pred.csv --iou 0.5 -o results.csv
        """
    )
    parser.add_argument(
        "gt_csv",
        type=str,
        help="Ground Truth CSVファイルパス"
    )
    parser.add_argument(
        "pred_csv",
        type=str,
        help="予測結果CSVファイルパス"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="マッチング用IoU閾値 (default: 0.5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="結果をCSVファイルに保存"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="詳細表示を抑制"
    )
    
    args = parser.parse_args()
    
    # 評価実行
    results = run_evaluation(
        gt_csv=args.gt_csv,
        pred_csv=args.pred_csv,
        iou_threshold=args.iou,
        print_details=not args.quiet
    )
    
    # 結果表示
    print_results(results)
    
    # CSV保存
    if args.output:
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()