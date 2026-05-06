import cv2
import numpy as np
from ultralytics import YOLO
from skimage.feature import local_binary_pattern
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

class OnlineGMM2:
    def __init__(self, dim, threshold=0.1, max_components=50, init_covariance=1.0):
        """
        dim: データ次元
        threshold: 既存コンポーネントの最大責任度がこの値未満なら新規コンポーネントを追加
        max_components: 最大コンポーネント数
        init_covariance: 新規コンポーネントの初期共分散（対角行列のスケール）
        """
        self.dim = dim
        self.threshold = threshold
        self.max_components = max_components
        self.init_cov = init_covariance

        # コンポーネントごとのパラメータ
        self.means = []          # 平均ベクトルのリスト
        self.covariances = []    # 共分散行列のリスト
        self.weights = []        # 混合係数（未正規化）
        self.Nk = []             # 各コンポーネントの「仮想観測数」
        self.total_N = 0         # 全サンプル数の累積（仮想観測数の合計）

    def partial_fit(self, x):
        """
        1サンプル x を受け取ってパラメータ更新。
        x: shape=(dim,)
        """
        x = np.asarray(x).reshape(-1)
        # 初回サンプル時は必ず新規コンポーネントを作成
        if not self.means:
            self._add_component(x)
            return

        # 既存コンポーネントの責任度を計算
        probs = np.array([
            w * multivariate_normal.pdf(x, mean=mu, cov=Sigma)
            for w, mu, Sigma in zip(self.weights, self.means, self.covariances)
        ])
        total_prob = probs.sum()
        if total_prob > 0:
            resp = probs / total_prob
        else:
            # 全て underflow した場合は均等割り
            resp = np.ones(len(self.means)) / len(self.means)

        # いちばん高い責任度がしきい値未満か
        if resp.max() < self.threshold and len(self.means) < self.max_components:
            self._add_component(x)
            return

        # オンラインEM の更新式
        for k, r in enumerate(resp):
            Nk_new = self.Nk[k] + r
            lr = r / Nk_new   # 学習率
            mu_old = self.means[k]
            # 平均の更新
            mu_new = mu_old + lr * (x - mu_old)
            # 共分散の更新
            diff = (x - mu_new).reshape(-1, 1)
            Sigma_old = self.covariances[k]
            Sigma_new = Sigma_old + lr * (diff @ diff.T - Sigma_old)

            self.means[k] = mu_new
            self.covariances[k] = Sigma_new
            self.Nk[k] = Nk_new

        self.total_N += 1
        # 混合係数を再計算（正規化は任意）
        self.weights = [nk / self.total_N for nk in self.Nk]

    def _add_component(self, x):
        """ 新規コンポーネントを追加 """
        self.means.append(x.copy())
        self.covariances.append(np.eye(self.dim) * self.init_cov)
        # 初期「観測数」は1、全体にも1加算
        self.Nk.append(1.0)
        self.total_N += 1
        # 重みも更新
        self.weights = [nk / self.total_N for nk in self.Nk]

    def predict_proba(self, x):
        """ 各コンポーネントの責任度を返す """
        x = np.asarray(x).reshape(-1)
        probs = np.array([
            w * multivariate_normal.pdf(x, mean=mu, cov=Sigma)
            for w, mu, Sigma in zip(self.weights, self.means, self.covariances)
        ])
        total = probs.sum()
        if total > 0:
            print(probs / total)
            return probs / total
        else:
            print(np.ones(len(self.means)) / len(self.means))
            return np.ones(len(self.means)) / len(self.means)

    def score_samples(self, X):
        """ 各サンプルの混合モデルによる対数尤度を計算 """
        X = np.atleast_2d(X)
        log_liks = []
        for x in X:
            probs = [
                w * multivariate_normal.pdf(x, mean=mu, cov=Sigma)
                for w, mu, Sigma in zip(self.weights, self.means, self.covariances)
            ]
            log_liks.append(np.log(np.sum(probs) + 1e-12))
        print('log_like',log_liks)
        return np.array(log_liks)

class OnlineGMM:
    def __init__(self, n_components, n_features, lr=0.05):
        self.K = n_components
        self.D = n_features
        self.lr = lr  # learning rate（更新速度）

        # 混合比（重み）を一様に初期化
        self.weights = np.ones(self.K) / self.K

        # 各ガウス分布の平均をランダム初期化
        self.means = np.random.rand(self.K, self.D)

        # 各ガウス分布の共分散行列を単位行列で初期化
        self.covariances = np.array([np.eye(self.D) for _ in range(self.K)])

    def update(self, x):
        # 各コンポーネントに対する所属確率（Eステップ）
        resp = np.array([
            self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k])
            for k in range(self.K)
        ])
        
        resp_sum = resp.sum()
        resp /= resp_sum + 1e-8  # 数値安定性

        # 各コンポーネントのパラメータを逐次更新（Mステップ）
        for k in range(self.K):
            r = resp[k]

            # 混合比の更新
            self.weights[k] = (1 - self.lr) * self.weights[k] + self.lr * r

            # 平均ベクトルの更新
            self.means[k] = (1 - self.lr) * self.means[k] + self.lr * r * x

            # 共分散行列の更新
            diff = (x - self.means[k]).reshape(-1, 1)
            self.covariances[k] = (1 - self.lr) * self.covariances[k] + \
                                  self.lr * r * (diff @ diff.T)

        # 正規化（混合比の和が1になるよう調整）
        self.weights /= np.sum(self.weights)

    def predict(self, x):
        # データがどのコンポーネントに属するかの予測
        probs = np.array([
            self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k])
            for k in range(self.K)
        ])
        print(probs)
        return np.argmax(probs)

def get_vector(mask_img):
    # Convert mask image to HSV color space
    hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    if hsv.shape[0] <= 3 or hsv.shape[1] <= 3:
        return None
    if hsv.shape[2] != 3:
        return None
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # Calculate histograms for H, S, and V channels
    h_hist = np.histogram(hsv[:,0], bins=16, range=(0,180), density=True)[0]
    s_hist = np.histogram(hsv[:,1], bins=16, range=(0,255), density=True)[0]
    v_hist = np.histogram(hsv[:,2], bins=16, range=(0,255), density=True)[0]

    hsv_feature = np.concatenate([h_hist, s_hist, v_hist])

    radius = 1       # 周囲の半径
    n_points = 8     # 半径に沿ってのポイント数
    METHOD = 'uniform'

    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    n_bins = n_points + 2
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([hsv_feature, lbp_hist])

def get_data(img, results):
    keypoints = results[0].keypoints.xy.numpy()
    data_list = []
    for keypoint in keypoints:
        x_array = np.array([keypoint[6, 0], keypoint[7, 0], keypoint[11,0], keypoint[12,0]])
        y_array = np.array([keypoint[6, 1], keypoint[7, 1], keypoint[11,1], keypoint[12,1]])
        #delete 0
        x_array = x_array[x_array > 0]
        y_array = y_array[y_array > 0]
        if x_array.size == 0 or y_array.size == 0:
            continue
        min_x = int(np.min(x_array))
        max_x = int(np.max(x_array))
        min_y = int(np.min(y_array))
        max_y = int(np.max(y_array))
        if min_x == max_x or min_y == max_y:
            continue
        mask_img = img[min_y:max_y, min_x:max_x]
        feature = get_vector(mask_img)
        if feature is not None:
            data_list.append(feature)
        # plt.figure(figsize=(10, 5))
        # plt.plot(feature, marker='o')
        # plt.title('Feature Vector')
        # plt.xlabel('Feature Index')
        # plt.ylabel('Feature Value')
        # plt.grid(True)
        # plt.savefig(f'feature_vector{count}.png')
        # cv2.imwrite(f'cloth{count}.jpg', mask_img)
    return data_list

# def get(img, results, index):
#     keypoints = results[0].keypoints.xy.numpy()
#     keypoint = keypoints[index]
#     x_array = np.array([keypoint[6, 0], keypoint[7, 0], keypoint[11,0], keypoint[12,0]])
#     y_array = np.array([keypoint[6, 1], keypoint[7, 1], keypoint[11,1], keypoint[12,1]])
#     #delete 0
#     x_array = x_array[x_array > 0]
#     y_array = y_array[y_array > 0]
#     if x_array.size == 0 or y_array.size == 0:
#         return None
#     min_x = int(np.min(x_array))
#     max_x = int(np.max(x_array))
#     min_y = int(np.min(y_array))
#     max_y = int(np.max(y_array))
#     if min_x == max_x or min_y == max_y:
#         return None
#     mask_img = img[min_y:max_y, min_x:max_x]
#     feature = get_vector(mask_img)
#     print(feature)
#     return feature

def get(img, results, index):
    keypoints = results[0].boxes.xyxy.numpy()
    keypoint = keypoints[index]
    x_array = np.array([keypoint[0], keypoint[2]])
    y_array = np.array([keypoint[1], keypoint[3]])
    #delete 0
    x_array = x_array[x_array > 0]
    y_array = y_array[y_array > 0]
    if x_array.size == 0 or y_array.size == 0:
        return None
    min_x = int(np.min(x_array))
    max_x = int(np.max(x_array))
    min_y = int(np.min(y_array))
    max_y = int(np.max(y_array))
    if min_x == max_x or min_y == max_y:
        return None
    mask_img = img[min_y:max_y, min_x:max_x]
    feature = get_vector(mask_img)
    return feature
    
def sim_cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


def main():
    model = YOLO('yolo11s-pose.pt')
    count = 0
    im1 = cv2.imread('test1.jpg')
    im2 = cv2.imread('test2.jpg')

    results1 = model(im1)
    results2 = model(im2)

    list1 = get_data(im1, results1)
    list2 = get_data(im2, results2)

    start = time.time()
    gmm = OnlineGMM(n_components=4, n_features=len(list1[0]))

    for i in range(len(list1)):
        gmm.update(list1[i])

    for i in range(len(list2)):
        gmm.update(list2[i])

    for i in range(len(list1)):
        print(f"Image 1, Feature {i}: {gmm.predict(list1[i])}")
    for i in range(len(list2)):
        print(f"Image 2, Feature {i}: {gmm.predict(list2[i])}")

    print("Time taken:", time.time() - start)

    cv2.imwrite('result1.jpg', results1[0].plot())
    cv2.imwrite('result2.jpg', results2[0].plot())

if __name__ == "__main__":
    main()