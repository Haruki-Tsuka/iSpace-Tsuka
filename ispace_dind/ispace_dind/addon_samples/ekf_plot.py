from ispace_dind.addons.addon_base import AddonBase, addon
from ispace_dind.utils.event_handler import Event

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np
import copy

@addon
class TrackAnime(AddonBase):

    FRAME_AMOUNT = 400

    def __init__(self, node):
        self.node = node
        self.id_colors = {}
        self.color_map = plt.cm.get_cmap('tab10')
        self.fig, self.ax = plt.subplots()
        self.count = 0
        self.tracking_data = []

    def register(self):
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.save_track)

    def save_track(self, tracker_list):
        if self.FRAME_AMOUNT > self.count:
            self.tracking_data.append(copy.deepcopy(tracker_list))
            if len(tracker_list) > 0:
                print(tracker_list[0].ekf)
        elif self.count == self.FRAME_AMOUNT:
            ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.tracking_data), init_func=self.init, blit=False, interval=200, repeat=False)
            ani.save('tracking_animation.mp4', writer='ffmpeg', fps=5)
            print('==================')
            print('GENERATED MOVIE')
            print('==================')
        self.count += 1
        print(self.count)

    def get_color_for_id(self, obj_id):
        if obj_id not in self.id_colors:
            self.id_colors[obj_id] = self.color_map(len(self.id_colors) % 10)
        return self.id_colors[obj_id]

    # 楕円（共分散から）をプロットする関数
    def get_cov_ellipse(self, mean, cov, n_std=2.0):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        return Ellipse(xy=mean, width=width, height=height, angle=angle, alpha=0.3)
    
    def init(self):
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("Extended Kalman Filter Tracking")
        return []

    def update(self, frame_idx):
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        frame = self.tracking_data[frame_idx]
        if len(frame) == 0:
            return []

        self.ax.set_title(f"Time {frame[0].last_predict:.2f}")

        for obj in frame:
            obj_id = obj.tracker_id
            mean = obj.ekf.x[0:2]
            print(f'obj {obj.tracker_id}, mean {mean}')
            cov = obj.pred_P[0:2, 0:2]
            if obj_id < 0:
                continue
            color = self.get_color_for_id(obj_id)

            self.ax.plot(*mean, marker='o', color=color, label=f"ID {obj_id}")
            ellipse = self.get_cov_ellipse(mean, cov)
            ellipse.set_facecolor(color)
            self.ax.add_patch(ellipse)

        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(loc='upper left')

        return []
