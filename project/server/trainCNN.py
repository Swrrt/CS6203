import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
import dataset
import imageFromUrl as iurl
train_path = '/home/ubuntu/project/data/train'
test_path = '/home/ubuntu/project/data/test'
catsUrlFile = 'cats.txt'
dogsUrlFile = 'dogs.txt'
image_size = 56
classes = ['cats','dogs'] 
#iurl.downloadImage(catsUrlFile,train_path+'/cats')
#iurl.downloadImage(dogsUrlFile,train_path+'/dogs')
X, Y, ids, cls, testX, testY, ids, cls= dataset.load_train(train_path,image_size,classes)
#testX, testY, ids, cls= dataset.load_train(test_path,image_size,classes)
X = X.reshape([-1, 56, 56, 1])
testX = testX.reshape([-1, 56, 56, 1])

print("Size is {}".format(X.size))
# Building convolutional network
network = input_data(shape=[None, 56, 56, 1], name='input')
network = conv_2d(network, 40, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
vetwork = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.05,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
print("start")
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
snapshot_step=100, show_metric=True, run_id='convnet_mnist')
print(model.predict(testX))
model.save("model.pkl")
model1 = model
model1.load("model.pkl")
print(model1.predict(X))

