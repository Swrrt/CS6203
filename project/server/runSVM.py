import cv2
import dataset
import imageFromUrl as iurl
import os
import glob
import random
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
def runSVM(path, image_size = 32, model_file = 'model.pkl', download = True):
	print('Predict with HOG feature and SVM')
	if download:
		print('Downloading images')
		iurl.download(urlfile, path)
	X, num = dataset.load_run(path, image_size, True)
	print("Number of data:{}".format(num))
	clf = joblib.load(model_file)
	result = clf.predict(X)
	print(result)

