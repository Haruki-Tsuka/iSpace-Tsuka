import pyrealsense2 as rs
import numpy as np
import os

# 保存先フォルダ
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# RealSense pipeline 作成
pipeline = rs.pipeline()
config = rs.config()

# 使用するストリームを設定
# 研究で使う画像サイズ・FPSに合わせることが重要
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # ストリーミング開始
    profile = pipeline.start(config)

    # Color stream の内部パラメータを取得
    color_profile = profile.get_stream(rs.stream.color)
    video_profile = color_profile.as_video_stream_profile()
    intr = video_profile.get_intrinsics()

    # OpenCV形式のカメラ行列 mtx
    mtx = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # OpenCV形式の歪み係数 dist
    # RealSenseのcoeffsは [k1, k2, p1, p2, k3] の形
    dist = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)

    print("mtx =")
    print(mtx)

    print("dist =")
    print(dist)

    print("dist shape:", dist.shape)

    # npy形式で保存
    np.save(os.path.join(save_dir, "mtx.npy"), mtx)
    np.save(os.path.join(save_dir, "dist.npy"), dist)

    print("保存完了:")
    print(os.path.join(save_dir, "mtx.npy"))
    print(os.path.join(save_dir, "dist.npy"))

finally:
    pipeline.stop()