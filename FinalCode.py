import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os

S = 0
#number of photos considered for  detection
n = 5

Wobj_mean1 = []
Wobj_max1 = []
Wobj_var1 = []
Wobj_mean2 = []
Wobj_max2 = []
Wobj_var2 = []
Wobj_mean3 = []
Wobj_max3 = []
Wobj_var3 = []
Wobj_mean4 = []
Wobj_max4 = []
Wobj_var4 = []
Wobj_mean5 = []
Wobj_max5 = []
Wobj_var5 = []
Wobj_obj = []

#address of data
path_obj = '../object/'
path_noObj = '../noObject/'
dir_num_obj = len([name for name in os.listdir(path_obj) if os.path.isdir(path_obj+name)])
dir_num_noObj = len([name for name in os.listdir(path_noObj) if os.path.isdir(path_noObj+name)])


for directory in range(1, dir_num_obj+1):
    # frame sequence analysis
    folder_path = path_obj + str(directory) + '/'
    img_num = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + name)])
    for i in range(1, img_num - (n - 1) + 1):
        data1 = cv2.imread(folder_path+str(i)+'.jpg', -1)
        data2 = cv2.imread(folder_path+str(i+1)+'.jpg', -1)
        data3 = cv2.imread(folder_path+str(i+2)+'.jpg', -1)
        data4 = cv2.imread(folder_path+str(i+3)+'.jpg', -1)
        data5 = cv2.imread(folder_path+str(i+4)+'.jpg', -1)
        Wobj_mean1.append(np.mean(data1))
        Wobj_max1.append(np.max(data1))
        Wobj_var1.append(np.var(data1) / 255)
        Wobj_mean2.append(np.mean(data2))
        Wobj_max2.append(np.max(data2))
        Wobj_var2.append(np.var(data2) / 255)
        Wobj_mean3.append(np.mean(data3))
        Wobj_max3.append(np.max(data3))
        Wobj_var3.append(np.var(data3) / 255)
        Wobj_mean4.append(np.mean(data4))
        Wobj_max4.append(np.max(data4))
        Wobj_var4.append(np.var(data4) / 255)
        Wobj_mean5.append(np.mean(data5))
        Wobj_max5.append(np.max(data5))
        Wobj_var5.append(np.var(data5) / 255)
        Wobj_obj.append(1)

for directory in range(1, dir_num_noObj + 1):
    # frame sequence analysis
    folder_path = path_noObj + str(directory) + '/'
    img_num = len([name for name in os.listdir(folder_path) if os.path.isfile(folder_path + name)])
    for i in range(1, img_num - (n - 1) + 1):
        data1 = cv2.imread(folder_path + str(i)+'.jpg', -1)
        data2 = cv2.imread(folder_path + str(i + 1)+'.jpg', -1)
        data3 = cv2.imread(folder_path + str(i + 2)+'.jpg', -1)
        data4 = cv2.imread(folder_path + str(i + 3)+'.jpg', -1)
        data5 = cv2.imread(folder_path + str(i + 4)+'.jpg', -1)
        Wobj_mean1.append(np.mean(data1))
        Wobj_max1.append(np.max(data1))
        Wobj_var1.append(np.var(data1) / 255)
        Wobj_mean2.append(np.mean(data2))
        Wobj_max2.append(np.max(data2))
        Wobj_var2.append(np.var(data2) / 255)
        Wobj_mean3.append(np.mean(data3))
        Wobj_max3.append(np.max(data3))
        Wobj_var3.append(np.var(data3) / 255)
        Wobj_mean4.append(np.mean(data4))
        Wobj_max4.append(np.max(data4))
        Wobj_var4.append(np.var(data4) / 255)
        Wobj_mean5.append(np.mean(data5))
        Wobj_max5.append(np.max(data5))
        Wobj_var5.append(np.var(data5) / 255)
        Wobj_obj.append(0)


feature = pd.DataFrame(
    {'Mean1': Wobj_mean1, 'Mean2': Wobj_mean2, 'Mean3': Wobj_mean3, 'Mean4': Wobj_mean4, 'Mean5': Wobj_mean5,
     'Max1': Wobj_max1, 'Max2': Wobj_max2, 'Max3': Wobj_max3, 'Max4': Wobj_max4, 'Max5': Wobj_max5,
     'Var1': Wobj_var1, 'Var2': Wobj_var2, 'Var3': Wobj_var3, 'Var4': Wobj_var4, 'Var5': Wobj_var5, 'Obj': Wobj_obj})

X = feature.loc[:, ['Mean1', 'Mean2', 'Mean3', 'Mean4', 'Mean5',
                    'Max1', 'Max2', 'Max3', 'Max4', 'Max5',
                    'Var1', 'Var2', 'Var3', 'Var4', 'Var5']]
Y = feature.loc[:, ['Obj']]



#learning model
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.9)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print('confusion matrix')
print(confusion_matrix(Y_test, Y_pred))
