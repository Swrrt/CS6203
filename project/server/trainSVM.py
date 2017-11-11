import cv2
import dataset
import imageFromUrl as iurl
import os
import glob
import random
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
def trainSVM(train_path, classes, image_size = 32, model_file = 'model.pkl', download_flag = False):
	print('Training with HOG feature and SVM')
	if download_flag:
		print('Downloading images')
		for i in classes:
			iurl.downloadImage(i+'.txt', train_path+'/'+i)
	X, Y, ids, cls, testX, testY, ids, cls, num = dataset.load_train(train_path, image_size, classes, 5, True)
	print ("Number of training data: {}".format(num))
	rand = np.random.RandomState(321)
	shuffle = rand.permutation(len(X))
#	print(X)
#	print(testX)
#	print(Y)
#	print(len(X[:-1]),len(Y))
	
	X = X[shuffle]
	Y = Y[shuffle]
#	print(X)
#	print(Y)
#	pause()
	clf = svm.SVC()	
	clf.fit(X, Y)
	result = clf.predict(testX)
	mask = result==testY
	correct = np.count_nonzero(mask)
#	print result
	print("Accuracy: {}",correct*100.0/len(result))
	joblib.dump(clf, model_file)
#train_path = '/home/ubuntu/data/train'
#image_size = 64
#classes = ['cats', 'dogs', 'insects', 'horses', 'goldfishes']
#classes = ['cat','dogs','insects', 'horses','goldfishes']
train_path = '/home/ubuntu/data/mnist_png/training'
image_size = 28
classes = ['0','1','2','3','4','5','6','7','8','9']
model_file = 'svm_model.pkl'
trainSVM(train_path, classes, image_size, model_file)
					

