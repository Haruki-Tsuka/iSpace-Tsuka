
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ID毎に一貫した色を生成するためのカラーマップ
def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """トラックIDに基づいて一貫した色を返す"""
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))


def draw_tracking_results(
    image: np.ndarray,
    detections: list[dict],
    timestamp: str
) -> np.ndarray:
    """
    トラッキング結果を画像に描画する
    
    Args:
        image: 描画対象の画像
        detections: 検出結果のリスト [{'id': int, 'x1': float, ...}, ...]
        timestamp: 現在のタイムスタンプ
    
    Returns:
        描画済みの画像
    """
    img = image.copy()
    
    # タイムスタンプを表示
    cv2.putText(
        img, f"Timestamp: {timestamp}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        img, f"Detections: {len(detections)}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    
    for det in detections:
        track_id = det['id']
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        
        # IDに基づいた色を取得
        color = get_color_for_id(track_id)
        
        # バウンディングボックスを描画
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # IDラベルの背景
        label = f"ID: {track_id}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color, -1
        )
        
        # IDラベルを描画
        cv2.putText(
            img, label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    
    return img


def get_sorted_images(image_dir: Path) -> list[Path]:
    """
    指定ディレクトリ内の画像ファイルをタイムスタンプ順にソートして返す
    """
    extensions = {'.jpg', '.jpeg', '.png'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    
    # ファイル名（拡張子除く）でソート
    # タイムスタンプ形式を想定しているため、文字列ソートで時系列順になる
    images.sort(key=lambda p: p.stem)
    
    return images


def run_tracking(
    image_dir: str,
    output_csv: str,
    model_name: str = "yolo11x.pt",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = "0",
    visualize: bool = False,
    display_scale: float = 1.0,
    wait_ms: int = 1
) -> None:
    """
    画像群に対してトラッキングを実行し、結果をCSVに保存する
    
    Args:
        image_dir: 入力画像が格納されているディレクトリパス
        output_csv: 出力CSVファイルパス
        model_name: YOLOモデル名（デフォルト: yolo11x.pt）
        conf_threshold: 検出信頼度閾値
        iou_threshold: NMS用IoU閾値
        device: 使用デバイス（"0"=GPU, "cpu"=CPU）
        visualize: トラッキング結果を表示するかどうか
        display_scale: 表示画像のスケール（1.0=原寸）
        wait_ms: 表示時の待機時間（ms）、0で手動送り
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # 画像ファイルを取得・ソート
    images = get_sorted_images(image_path)
    if not images:
        raise ValueError(f"No image files found in: {image_dir}")
    
    print(f"Found {len(images)} images in {image_dir}")
    
    # YOLOモデルをロード
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # 結果を格納するリスト
    results_data: list[dict] = []
    
    # 各画像に対してトラッキングを実行
    # persist=True でトラッカーの状態を維持
    print("Running tracking...")
    if visualize:
        print("Visualization enabled. Press 'q' to quit, 'space' to pause/resume.")
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    
    paused = False
    
    for i, img_path in enumerate(images):
        timestamp = img_path.stem  # 拡張子を除いたファイル名
        
        # トラッキング実行（classes=[0]で人物のみ検出）
        results = model.track(
            source=str(img_path),
            persist=True,
            classes=[0],  # person class only
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        
        # 現在フレームの検出結果を格納
        frame_detections: list[dict] = []
        
        # 結果を抽出
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes
            
            # トラッキングIDが存在する場合のみ処理
            if boxes.id is not None:
                for j, box in enumerate(boxes):
                    track_id = int(boxes.id[j].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        'timestamp': timestamp,
                        'id': track_id,
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    }
                    frame_detections.append(detection)
                    results_data.append(detection)
        
        # 可視化
        if visualize:
            img = cv2.imread(str(img_path))
            if img is not None:
                vis_img = draw_tracking_results(img, frame_detections, timestamp)
                
                # スケール調整
                if display_scale != 1.0:
                    new_w = int(vis_img.shape[1] * display_scale)
                    new_h = int(vis_img.shape[0] * display_scale)
                    vis_img = cv2.resize(vis_img, (new_w, new_h))
                
                cv2.imshow("Tracking", vis_img)
                
                # キー入力処理
                while True:
                    key = cv2.waitKey(wait_ms if not paused else 0) & 0xFF
                    
                    if key == ord('q'):
                        print("\nVisualization stopped by user.")
                        cv2.destroyAllWindows()
                        visualize = False
                        break
                    elif key == ord(' '):
                        paused = not paused
                        if paused:
                            print("Paused. Press 'space' to resume.")
                        else:
                            print("Resumed.")
                    elif key == ord('s'):
                        # 現在フレームを保存
                        save_path = f"frame_{timestamp}.png"
                        cv2.imwrite(save_path, vis_img)
                        print(f"Saved: {save_path}")
                    
                    if not paused:
                        break
        
        # 進捗表示
        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"Processed {i + 1}/{len(images)} images")
    
    if visualize:
        cv2.destroyAllWindows()
    
    # CSVに保存
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'id', 'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()
        writer.writerows(results_data)
    
    print(f"\nTracking complete!")
    print(f"Total detections: {len(results_data)}")
    print(f"Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11を使用した人物トラッキング",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_dir",
        type=str,
        default="img",
        help="入力画像が格納されているディレクトリパス"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tracking_results.csv",
        help="出力CSVファイルパス"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo11x.engine",
        help="YOLOモデル名（yolo11n/s/m/l/x）"
    )
    parser.add_argument(
        "-c", "--conf",
        type=float,
        default=0.5,
        help="検出信頼度閾値"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS用IoU閾値"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="0",
        help="使用デバイス（'0'=GPU, 'cpu'=CPU）"
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="トラッキング結果をリアルタイム表示"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="表示画像のスケール（1.0=原寸）"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=1,
        help="表示時の待機時間（ms）、0で手動送り"
    )
    
    args = parser.parse_args()
    
    run_tracking(
        image_dir=args.image_dir,
        output_csv=args.output,
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        visualize=args.visualize,
        display_scale=args.scale,
        wait_ms=args.wait
    )


if __name__ == "__main__":
    main()