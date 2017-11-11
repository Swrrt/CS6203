import tensorflow as tf
import imageFromUrl as iurl
import dataset
import numpy as np
train_path = '/home/ubuntu/data/train'
classes = ['cats', 'dogs']
image_size = 64
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10
num_input = image_size * image_size
num_classes = len(classes)
dropout = 0.75
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
def conv2d(x, W, b, strides = 1):
	x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)
def maxpool2d(x, k = 2):
	return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')
def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape = [-1, image_size, image_size, 1])
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k = 2)
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k = 2)
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out
weights = {
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
	'out': tf.Variable(tf.random_normal([1024, num_classes]))
}
biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	trainX, trainY, ids, cls, textX, testY, ids, cls, num = dataset.load_train(train_path, image_size, classes)
	tnum = (num - 1)/num_steps + 1
	tX = 0
	tY = 0
	for step in range(1, num_steps+1):
		ttX = tX + tnum * num_input
		ttY = tY + tnum * num_classes
		if ttX >= trainX.size : ttX = trainX.size 
		if ttY >= trainY.size : ttY = trainY.size 
		batch_X = trainX[tX:ttX]
		batch_X = np.reshape(batch_X, (-1, image_size*image_size))
		batch_Y = trainY[tY:ttY]
	
		tX = ttX 
		tY = ttY 
		sess.run(train_op, feed_dict={X: batch_X, Y:batch_Y, keep_prob: 0.8})
		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_X, Y: batch_Y, keep_prob: 1.0})
			print("Step " + str(step) + ", Minibatch Loss = " + "{:.4f}".format(loss) + ", Training Accuracy = " + "{:.3f}".format(acc))
	print("Optimization Finished!")
	print("Testing Accurarcy:", sess.run(accuracy, feed_dict = {X: testX, Y: testY, keep_prob: 1.0}))




