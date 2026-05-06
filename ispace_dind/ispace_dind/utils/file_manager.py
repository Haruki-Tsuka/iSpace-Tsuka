import os
import csv
import pandas as pd
import cv2
import configparser

class CSVFileManager:
    def __init__(self, dir=None, csv_name='data.csv', columns=None):
        if dir is None:
            self.dir = os.getcwd()
        else:
            self.dir = dir
        self.csv_name = csv_name
        self.csv_path = os.path.join(self.dir, self.csv_name)
        self.columns = columns

    def create(self):
        # ディレクトリが存在しない場合、作成
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        # CSVファイルが存在しない場合、作成
        if not os.path.exists(self.csv_path):
            if self.columns is None:
                raise ValueError("列名(columns)を指定してください。")
            else:
                with open(self.csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.columns)

    def add(self, data):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("CSVファイルが存在しません。create()メソッドを呼び出してください。")
        
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def clear(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        
        self.create()

    def get_data_as_list(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("CSVファイルが存在しません。create()メソッドを呼び出してください。")
        with open(self.csv_path, 'r') as file:
            reader = csv.reader(file)
            return [row for row in reader]
        
    def get_data_as_panda(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("CSVファイルが存在しません。create()メソッドを呼び出してください。")
        return pd.read_csv(self.csv_path)

class ImageFileManager:

    def __init__(self, dir=None, img_name='image', file_type='png'):
        if dir is None:
            self.dir = os.getcwd()
        else:
            self.dir = dir
        self.img_name = img_name
        self.file_type = file_type
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.index = 0
        img_names = os.listdir(self.dir)
        numeric_file_names = [name.split(".")[0] for name in img_names if name.endswith(".png") and name[:-4].isdigit()]
        if numeric_file_names:
            self.index = max(map(int, numeric_file_names))

    def add(self, img):
        self.index += 1
        path = os.path.join(self.dir, f'{self.img_name}_{self.index:05d}.{self.file_type}')
        cv2.imwrite(path, img)
        return self.index

# 使用例
if __name__ == "__main__":
    columns = ['Name', 'Age', 'Country']
    manager = CSVFileManager(columns=columns)
    
    manager.create()
    manager.add(['Alice', 25, 'USA'])
    manager.add(['Bob', 30, 'Canada'])
    
    print("CSVファイルの内容:")
    with open(manager.csv_path, 'r') as file:
        for line in file:
            print(line.strip())
    
    manager.clear()
    print("CSVファイルの内容 (クリア後):")
    with open(manager.csv_path, 'r') as file:
        for line in file:
            print(line.strip())

class ConfigManager:

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load()

    def load(self):
        self.config.read(self.config_file)

    def save(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def get(self, section, key, default=None):
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def set(self, section, key, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, value)
        self.save()

    def remove_section(self, section):
        if self.config.has_section(section):
            self.config.remove_section(section)
            self.save()

    def remove_option(self, section, key):
        if self.config.has_option(section, key):
            self.config.remove_option(section, key)
            self.save()
