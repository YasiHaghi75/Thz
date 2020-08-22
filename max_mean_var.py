# This Python script is used to build a Support Vector Machine (SVM) model based on the THz images dataset.
# The feature vector used to train SVM is obtained from concatenation maximum, mean, and variance of pixels
# in n consecutive images, where n is the number of images.
import argparse
from image_reader import build_dataset
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix


ap = argparse.ArgumentParser()
ap.add_argument('--object', help='path to the folder containing object-labeled images')
ap.add_argument('--noObject', help='path to the folder containing noObject-labeled images')
ap.add_argument('-n', help='number of frames in an evaluated sequence')
args = vars(ap.parse_args())

if args['n'] is None:
    n = 5
else:
    n = int(args['n'])

if args['object'] is None:
    path_obj = './thz_images/object/'
else:
    path_obj = args['object']

if args['noObject'] is None:
    path_noObj = './thz_images/noObject/'
else:
    path_noObj = args['noObject']

train_df, test_df = build_dataset(path_obj, path_noObj, n)

X_train = train_df.iloc[:, 0:n*3]
Y_train = train_df.iloc[:, -1]
Y_train = Y_train.astype(int)

X_test = test_df.iloc[:, 0:n*3]
Y_test = test_df.iloc[:, -1]
Y_test = Y_test.astype(int)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print('confusion matrix')
print(confusion_matrix(Y_test, Y_pred))
