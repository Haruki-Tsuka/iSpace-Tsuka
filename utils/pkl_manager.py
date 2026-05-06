import numpy as np
from utils.file_manager import CSVFileManager
import pickle
import os
from enum import Enum
import pandas as pd

class PklManager:

    def __init__(self, pkl_dir, pkl_name, csv_dir, csv_name, column_list):
        if pkl_dir is None:
            self.pkl_dir = os.getcwd()
        else:
            self.pkl_dir = pkl_dir
        self.pkl_name = pkl_name
        self.pkl_path = os.path.join(self.pkl_dir, self.pkl_name)
        self.csv_dir = csv_dir
        self.csv_name = csv_name
        self.csv_file_manager = CSVFileManager(dir=csv_dir, csv_name=csv_name, columns=column_list)
        self.csv_file_manager.create()
        self.model = self.__load_pkl()

    def clear_csv(self):
        self.csv_file_manager.clear()

    def get_bone(self, results, index=0):
        if results[0].keypoints.conf is None:
            return None
        xyn = results[0].keypoints.xyn.numpy()
        bones = []
        bones = self.__get_vector(xyn, index, 0, 1, bones)
        bones = self.__get_vector(xyn, index, 0, 2, bones)
        bones = self.__get_vector(xyn, index, 1, 2, bones)
        bones = self.__get_vector(xyn, index, 1, 3, bones)
        bones = self.__get_vector(xyn, index, 2, 4, bones)
        bones = self.__get_vector(xyn, index, 3, 5, bones)
        bones = self.__get_vector(xyn, index, 4, 6, bones)
        bones = self.__get_vector(xyn, index, 5, 6, bones)
        bones = self.__get_vector(xyn, index, 5, 7, bones)
        bones = self.__get_vector(xyn, index, 7, 9, bones)
        bones = self.__get_vector(xyn, index, 6, 8, bones)
        bones = self.__get_vector(xyn, index, 8, 10, bones)
        bones = self.__get_vector(xyn, index, 5, 11, bones)
        bones = self.__get_vector(xyn, index, 6, 12, bones)
        bones = self.__get_vector(xyn, index, 11, 12, bones)
        bones = self.__get_vector(xyn, index, 11, 13, bones)
        bones = self.__get_vector(xyn, index, 12, 14, bones)
        bones = self.__get_vector(xyn, index, 13, 15, bones)
        bones = self.__get_vector(xyn, index, 14, 16, bones)
        return bones

    def __get_vector(self, xyn, index, id1, id2, bones):
        bones.append(xyn[index, id1, 0] - xyn[index, id2, 0])
        bones.append(xyn[index, id1, 0] - xyn[index, id2, 0])
        return bones
    
    def save_to_csv(self, row):
        self.csv_file_manager.add(row)
        

    def __load_pkl(self):
        if not os.path.exists(self.pkl_path):
            return None
        with open(self.pkl_path, 'rb') as f0:
            return pickle.load(f0)
        
    def predict(self, row):
        if not self.model or not row:
            return None, None
        ROW_DATA = pd.DataFrame([row])
        action_class = self.model.predict(ROW_DATA)[0]
        action_prob = self.model.predict_proba(ROW_DATA)[0]
        action_prob = round(action_prob[np.argmax(action_prob)],2)
        return action_class, action_prob
    

class ActionPklManager(PklManager):

    def __init__(self, pkl_dir=None, pkl_name='action_3d.pkl', csv_dir=None, csv_name='action_3d.csv'):
        super().__init__(pkl_dir, pkl_name, csv_dir, csv_name, self.get_column_list())
        self.before_coords = None

    def get_data_row(self, results, rs_manager):
        if results[0].keypoints.conf is None:
            return None
        xyn = results[0].keypoints.xyn.numpy()
        xyvn = np.column_stack((xyn[0],results[0].keypoints.conf.numpy()[0]))
        xy = results[0].keypoints.xy.numpy()
        coords_s = list(xyvn.flatten())
        distance_s = [0 for i in range(34)]
        human_z = rs_manager.get_distance(xy[0,0,0], xy[0,0,1])
        if human_z == 0:
            return None
        rpj = 0
        for rpi in range(len(coords_s)):
            if rpi % 3 != 2:
                if self.before_coords is not None:
                    distance_s[rpj] = coords_s[rpi] - self.before_coords[rpi]
                    rpj += 1
                if rpi > 1:
                    coords_s[rpi] = coords_s[rpi] - coords_s[rpi%3]
        coords_human = [round(xy[0,0,0]),round(xy[0,0,1]), round(human_z*100)]
        row = coords_s+coords_human+distance_s
        self.before_coords = list(xyvn.flatten())
        return row
    
    def get_data_row2(self, before_person, after_person):
        before_xyvn = np.column_stack((before_person.xyn, before_person.conf))
        after_xyvn = np.column_stack((after_person.xyn, after_person.conf))
        distance_s = after_xyvn - before_xyvn
        distance_s = list(distance_s.flatten())
        coords_s = list(after_xyvn.flatten())
        coords_human = [0, 0, 0]
        time_diff = after_person.timestamp - before_person.timestamp
        coords_human[0] = (after_person.coords_person[0] - before_person.coords_person[0]) / time_diff
        coords_human[1] = (after_person.coords_person[1] - before_person.coords_person[1]) / time_diff
        coords_human[2] = (after_person.coords_person[2] - before_person.coords_person[2]) / time_diff
        row = coords_s+coords_human+distance_s
        return row

    def get_column_list(self):
        column = ['class']
        for i in range(17):
            column.append(f'skeleton_x{i}')
            column.append(f'skeleton_y{i}')
            column.append(f'skeleton_v{i}')
        column.append('human_x')
        column.append('human_y')
        column.append('human_z')
        for i in range(17):
            column.append(f'distance_s_x{i}')
            column.append(f'distance_s_y{i}')
            column.append(f'distance_s_v{i}')
        return column

class InteractType(Enum):
        BOOK = ['book_3d.pkl', 'book_3d.csv', 73]
        BOTTLE = ['bottle_3d.pkl', 'bottle_3d.csv', 39]

class InteractionPklManager(PklManager):
    def __init__(self, interactType:InteractType=None):
        if interactType:
            pkl_name = interactType.value[0]
            csv_name = interactType.value[1]
            self.ids = interactType.value[2]
            self.THRESHOLD = 1.5
            self.type = interactType
        super().__init__(None, pkl_name, None, csv_name, self.get_column_list())
    
    def get_data_row(self, before_person, after_person, obj):
        before_xyvn = np.column_stack((before_person.xyn, before_person.conf))
        after_xyvn = np.column_stack((after_person.xyn, after_person.conf))
        distance_s = after_xyvn - before_xyvn
        distance_s = list(distance_s.flatten())
        coords_s = list(after_xyvn.flatten())
        coords_human = [0, 0, 0]
        target_objs = obj[obj[:,0] == self.ids]
        if target_objs.shape[0] == 0:
            return None
        distances = np.linalg.norm(target_objs[:,5:] - after_person.coords_3d, axis=1)
        closest_index = np.argmin(distances)
        closest_obj = target_objs[closest_index]
        if distances[closest_index] > self.THRESHOLD:
            return None
        distance_so = np.linalg.norm(after_person.xyn[:] - closest_obj[1:3], axis=1)
        distance_so = list(distance_so.flatten())
        time_diff = after_person.timestamp - before_person.timestamp
        coords_human[0] = (after_person.coords_person[0] - before_person.coords_person[0]) / time_diff
        coords_human[1] = (after_person.coords_person[1] - before_person.coords_person[1]) / time_diff
        coords_human[2] = (after_person.coords_person[2] - before_person.coords_person[2]) / time_diff
        row = coords_s+coords_human+list(closest_obj[1:5])+[closest_obj[7]]+distance_so+distance_s
        return row

    def get_column_list(self):
        column = ['class']
        for i in range(17):
            column.append(f'skeleton_x{i}')
            column.append(f'skeleton_y{i}')
            column.append(f'skeleton_v{i}')
        column.append('human_x')
        column.append('human_y')
        column.append('human_z')
        column.append('obj_x')
        column.append('obj_y')
        column.append('obj_w')
        column.append('obj_h')
        column.append('obj_z')
        for i in range(17):
            column.append(f'distance_so{i}')
        for i in range(17):
            column.append(f'distance_s_x{i}')
            column.append(f'distance_s_y{i}')
            column.append(f'distance_s_v{i}')
        return column