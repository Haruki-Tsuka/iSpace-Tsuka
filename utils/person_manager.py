import numpy as np
import math
import time    

class Person:

    def __init__(self, xy, xyn, xyxyn, conf, b_conf, b_xyxy, rs_manager):
        self.xy = xy
        self.xyn = xyn
        self.xyxyn = xyxyn
        self.conf = conf
        self.b_conf = b_conf
        self.b_xyxy = b_xyxy
        mask = xy[:5,0] != 0
        self.coords_3d = None
        if np.any(mask):
            self.ex = int(np.mean(xy[:5,0][mask]))
            self.ey = int(np.mean(xy[:5,1][mask]))
        else:
            index = np.argmax(conf[5:])
            self.ex = int(xy[index,0])
            self.ey = int(xy[index,1])
        if self.ex < 10 or self.ex > rs_manager.width-10:
            self.coords_3d = None
        else:
            self.coords_3d = rs_manager.get_3d_coordinate(self.ex, self.ey)
        self.coords_person = None
        if self.coords_3d is not None:
            self.coords_3d[2] += 0.16   #人の頭の大きさを考慮し、距離を加算
            self.coords_person = [self.coords_3d[0]*10,self.coords_3d[1]*10, self.coords_3d[2]*10]
        self.bones = self.get_norm_bones(xyn)
        self.coods_s = self.get_coods_s(xyn, conf)

        self.id = 0
        self.gen = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.sim = 0
        self.sort_id=-1
        self.timestamp = time.time()


    def get_row(self):
        return self.bones + self.coords_person

    def get_coods_s(self, xyn, conf):
        xyvn = np.column_stack((xyn, conf))
        coords_s = list(xyvn.flatten())
        for rpi in range(len(coords_s)):
            if rpi % 3 != 2:
                if rpi > 1:
                    coords_s[rpi] = coords_s[rpi] - coords_s[rpi%3]
        return coords_s

    def get_bones(self, xyn):
        bones = []
        for i in range(17):
            bones = self.__get_vector(xyn, i, bones)
        return bones
    
    def get_similarity(self, another_person):
        list1 = np.array(self.bones)
        list2 = np.array(another_person.bones)

        zero_columns = (list1!=0) & (list2!=0)

        norm1 = np.linalg.norm(list1[zero_columns])
        norm2 = np.linalg.norm(list2[zero_columns])
        # ゼロ除算を防ぐために長さが0の場合は類似度を0とする
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 内積を計算
        dot_product = np.dot(list1[zero_columns], list2[zero_columns])
        # コサイン類似度を計算
        return (dot_product / (norm1 * norm2))

    
    def __get_vector(self, xyn, id, bones):
        x = (xyn[5,0]+xyn[6,0]+xyn[11,0]+xyn[12,0])/4 - xyn[id, 0]
        y = (xyn[5,1]+xyn[6,1]+xyn[11,1]+xyn[12,1])/4 - xyn[id, 1]
        if xyn[id, 0] == 0:
            x = 0
            y = 0
        bones.append(x)
        bones.append(y)
        return bones

class PersonManager:

    def __init__(self):
        self.before_list = []
        self.id = 0

    def get_distance(self, person, person2):
        x = (person.coords_3d[0] - person2.coords_3d[0])**2
        y = (person.coords_3d[1] - person2.coords_3d[1])**2
        z = (person.coords_3d[2] - person2.coords_3d[2])**2
        return math.sqrt(x+y+z)

    def get_person_list(self, results, rs_manager):
        person_list = []
        if results[0].keypoints.conf is None:
            return person_list
        conf_list = results[0].keypoints.conf.numpy()
        xyn_list = results[0].keypoints.xyn.numpy()
        xy_list = results[0].keypoints.xy.numpy()
        boxes = results[0].boxes.xyxyn.numpy()
        b_conf = results[0].boxes.conf
        b_xyxy = results[0].boxes.xyxy.numpy()
        for xy, xyn, conf, box, xyxy, bconf in zip(xy_list, xyn_list, conf_list, boxes, b_xyxy, b_conf):
            person = Person(xy, xyn, box, conf, bconf, xyxy, rs_manager)
            if person.coords_3d is None:
                continue
            for tmp in person_list:
                #重複検知
                if np.sqrt(np.sum((np.array(tmp.coords_3d) - np.array(person.coords_3d))**2)) < 0.3:
                    person.coords_person = None
                    break
            if person.coords_person is not None:
                person_list.append(person)
        return person_list
    