import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import dataset
import imageFromUrl as iurl
def runCNN(download = True, urlfile = 'url.txt', path = '/home/ubuntu/data/download', image_size = 32, model_file = 'model.pkl'):
	if download:
		iurl.downloadImage(urlfile, path)
	X, num = dataset.load_run(path,image_size)
	X = X.reshape([-1, image_size, image_size, 1])
	network = input_data(shape=[None, image_size, image_size, 1], name='input')
        network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = fully_connected(network, 32, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 64, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.0005, loss='categorical_crossentropy', name='target')
	model = tflearn.DNN(network)
	model.load(model_file)
	Y = model.predict(X)
	print (Y)
url = 'minist_test.txt'
path = '/home/ubuntu/data/download/'
image_size = 28
model = '/home/ubuntu/CS6203/project/server/minist_model.pkl'
runCNN(True, url, path, image_size, model)
