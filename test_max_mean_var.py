# This Python script gets the trained model and scalar and also the path to images of a trial as input and predicts
# whether the subject has any metallic object or not 
import argparse
import pickle
import cv2
import os
import numpy as np
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument('--model', help='path to SVM model')
ap.add_argument('--trial', help='path to trial directory')
ap.add_argument('--scalar', help='scalar used in the training data')
ap.add_argument('-n', help='number of frames in a sequence')
args = vars(ap.parse_args())

if args['n'] is None:
    n = 5
else:
    n = args['n']

model = pickle.load(open(args['model'], 'rb'))
scalar = pickle.load(open(args['scalar'], 'rb'))

folder_path = args['trial']
img_num = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + name)])
step = n

col_names = []
for i in range(n):
    col_names.append('max' + str(i + 1))
    col_names.append('mean' + str(i + 1))
    col_names.append('var' + str(i + 1))
test_df = pd.DataFrame(columns=col_names)
for i in range(1, img_num - (n - 1), step):
    feature_vec = []
    for idx in range(n):
        data = cv2.imread(folder_path + str(i + idx) + '.jpg', -1)
        feature_vec.append(np.max(data))
        feature_vec.append(np.mean(data))
        feature_vec.append(np.var(data))
    df = pd.DataFrame([feature_vec], columns=col_names)
    test_df = test_df.append(df, ignore_index=True)

test_df = scalar.transform(test_df)
y_pred = model.predict(test_df)
is_obj_certainty = 0
for label in y_pred:
    if label == 1:
        is_obj_certainty += 1

if is_obj_certainty / len(y_pred) >= 0.5:
    print('This Person Has a Metallic Object!')
else:
    print('This Person Is Safe')

