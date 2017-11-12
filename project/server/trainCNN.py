import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
import dataset
import imageFromUrl as iurl
def trainCNN(train_path, classes, image_size = 32,  model_file = 'model.pkl', download_flag = False):
	if download_flag:
		for i in classes:
			iurl.downloadImage(i+'.txt', train_path+'/'+i)
	
	#iurl.downloadImage(catsUrlFile,train_path+'/cats')
	#iurl.downloadImage(dogsUrlFile,train_path+'/dogs')
	X, Y, ids, cls, testX, testY, ids, cls, num= dataset.load_train(train_path,image_size,classes,5)
	#testX, testY, ids, cls= dataset.load_train(test_path,image_size,classes)
	print('Number of test data:{}'.format(num))
	X = X.reshape([-1, image_size, image_size, 1])
	testX = testX.reshape([-1, image_size, image_size, 1])
	# Building convolutional network
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
	network = fully_connected(network, len(classes), activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	# Training
	model = tflearn.DNN(network, tensorboard_verbose=0)
	print("start")
	model.fit({'input': X}, {'target': Y}, n_epoch=20, shuffle = True, batch_size = 100, validation_set=({'input': testX}, {'target': testY}), snapshot_step=100, show_metric=True, run_id='convnet_mnist')
	print(model.predict(testX))
	model.save(model_file)

#train_path = '/home/ubuntu/data/train'
#UrlFiles = ['cats.txt','dogs.txt']
#image_size = 64
#classes = ['cats', 'dogs', 'insects', 'horses', 'goldfishes']
train_path = '/home/ubuntu/data/mnist_png/training'
image_size = 28
classes = ['0','1','2','3','4','5','6','7','8','9']
model_file = 'minist_model.pkl'
trainCNN(train_path, classes, image_size, model_file)
