# This file contains the method "build_dataset" which builds two pandas dataframe from the THz images, one for training
# the model and one for testing. "build_dataset" gets the paths for object and noObject directories and the number of
# consecutive images in a sequence.
import os
import cv2
import numpy as np
import pandas as pd


def build_dataset(path_obj, path_noObj, n):
    dir_num_obj = len([name for name in os.listdir(path_obj) if os.path.isdir(path_obj + name)])
    dir_num_noObj = len([name for name in os.listdir(path_noObj) if os.path.isdir(path_noObj + name)])
    train_list_obj = [8, 13, 14, 15, 16]  # directory ID of object images used for training
    train_list_noObj = [1, 5]  # directory ID of noObject images used for training
    step = n

    col_names = []
    for i in range(n):
        col_names.append('max' + str(i + 1))
        col_names.append('mean' + str(i + 1))
        col_names.append('var' + str(i + 1))
    col_names.append('label')
    train_df = pd.DataFrame(columns=col_names)
    test_df = pd.DataFrame(columns=col_names)

    for directory in range(1, dir_num_obj+1):
        folder_path = path_obj + str(directory) + '/'
        img_num = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + name)])
        for i in range(1, img_num - (n - 1), step):
            feature_vec = []
            for idx in range(n):
                data = cv2.imread(folder_path + str(i + idx) + '.jpg', -1)
                feature_vec.append(np.max(data))
                feature_vec.append(np.mean(data))
                feature_vec.append(np.var(data))
            feature_vec.append(1)
            df = pd.DataFrame([feature_vec], columns=col_names)
            if directory in train_list_obj:
                train_df = train_df.append(df, ignore_index=True)
            else:
                test_df = test_df.append(df, ignore_index=True)

    for directory in range(1, dir_num_noObj+1):
        folder_path = path_noObj + str(directory) + '/'
        img_num = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + name)])
        for i in range(1, img_num - (n - 1), step):
            feature_vec = []
            for idx in range(n):
                data = cv2.imread(folder_path + str(i + idx) + '.jpg', -1)
                feature_vec.append(np.max(data))
                feature_vec.append(np.mean(data))
                feature_vec.append(np.var(data))
            feature_vec.append(0)
            df = pd.DataFrame([feature_vec], columns=col_names)
            if directory in train_list_noObj:
                train_df = train_df.append(df, ignore_index=True)
            else:
                test_df = test_df.append(df, ignore_index=True)

    return train_df, test_df
