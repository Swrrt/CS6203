import os
import glob
import numpy as np
import cv2
import random
import sys

def hog(img):    
	bin_n = 8
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist

def load_train(train_path, image_size, classes, valid_patio, feature = False):
	images = []
	labels = []
	v_images = []
	v_labels = []
	ids = []
	cls = []
	v_ids = []
	v_cls = []
	print('Reading training images')
	labeln = 0
	num = 0
	for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
    		index = classes.index(fld)
        	path = os.path.join(train_path, fld, '*')
	        print('Loading {} files (Index: {}) from {}'.format(fld, index,path))
        	files = glob.glob(path)
		labeln = labeln + 1
        	for fl in files:
			print('{}'.format(fl))
		        try:
#			for i in (1,1):
				image = cv2.imread(fl)
				image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
				if random.randint(0,99) > valid_patio :
					print('in trainset')
					if feature:
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
						
						hist = hog(image)
						images.append(hist)
						label = index
		    			else:
						image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
						images.append(image)
			            		label = np.zeros(len(classes))
		        			label[index] = 1.0
		        		labels.append(label)
		        		flbase = os.path.basename(fl)
			        	ids.append(flbase)
	        			cls.append(fld)
					num += 1
				else :
					print('in validation set')
					if feature:
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
						hist = hog(image)
						v_images.append(hist)
						label = index
		    			else:
						image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
						v_images.append(image)
			            		label = np.zeros(len(classes))
		        			label[index] = 1.0
		        		v_labels.append(label)
		        		flbase = os.path.basename(fl)
			        	v_ids.append(flbase)
	        			v_cls.append(fld)
			except:
				print('fail to load image')

	if not feature : 
		images = np.array(images)
		v_images = np.array(v_images)
	else : 
		images = np.float32(images)
		v_images = np.float32(v_images)
	labels = np.array(labels)
	ids = np.array(ids)
	cls = np.array(cls)
	v_labels = np.array(v_labels)
	v_ids = np.array(v_ids)
	v_cls = np.array(v_cls)
#	print("Num of images: {}".format(labels.size))
	return images, labels, ids, cls, v_images, v_labels, v_ids, v_cls, num

def load_run(run_path, image_size, feature = False):
	images = []
	num = 0
        path = os.path.join(run_path, '*')
	print('Loading run files from {}'.format(path))
       	files = glob.glob(path)
       	for fl in files:
		print('{}'.format(fl))
	        try:
			image = cv2.imread(fl)
			image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
			if feature:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				hist = hog(image)
				images.append(hist)
				num += 1
    			else:
				image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
				images.append(image)
				num += 1
		except:
			print('fail to load image')

	if not feature : 
		images = np.array(images)
	else : 
		images = np.float32(images)
	return images, num;
#
def load_test(test_path, image_size):
	path = os.path.join(test_path, '*g')
	files = sorted(glob.glob(path))
	X_test = []
	X_test_id = []
	print("Reading test images")
	for fl in files:
      		flbase = os.path.basename(fl)
		try:
			img = cv2.imread(fl)
      			img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
      			X_test.append(img)
	      		X_test_id.append(flbase)
		except:
			print('Fail to load image')
  ### because we're not creating a DataSet object for the test images, normalization happens here
	X_test = np.array(X_test, dtype=np.uint8)
	X_test = X_test.astype('float32')
	X_test = X_test / 255
  	return X_test, X_test_id

class DataSet(object):

  def __init__(self, images, labels, ids, cls):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    self._num_examples = images.shape[0]


    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_train(train_path, image_size, classes)
#  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

  return data_sets


def read_test_set(test_path, image_size):
  images, ids  = load_test(test_path, image_size)
  return images, ids
