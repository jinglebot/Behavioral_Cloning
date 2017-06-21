import csv
import cv2
import numpy as np
import sklearn

# open/read csv log file
samples = []
with open('./data/driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
	print('Number of lines on driving log: ', len(samples))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# image processing
def process_image(image):
	# crop image
	img = image[70:-25, :, :]

	# resize image
	img = cv2.resize(img, (64,64))

	# change colorspace
	img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	randbrightness = .2 + np.random.uniform()
	img[:,:,2] = img[:,:,2] * randbrightness
	img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

	# normalize image
	image = (img / 127.5) - 1.0

	return image

# fit_generator function
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			# get images and steering angles
			images = []
			measurements = []
			for batch_sample in batch_samples:
				for j in range(3):
					# open/read image files
					source_path = batch_sample[j].split('\\')[-1]
					filename = source_path.split('/')[-1]
					local_path = './data/IMG/' + filename
					image = cv2.imread(local_path)
					image = process_image(image)
					images.append(image)
				correction = 0.2
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				measurements.append(measurement + correction)
				measurements.append(measurement - correction)

			augmented_images = []
			augmented_measurements = []
			# augment data with flipped version of images and angles
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				flipped_image = cv2.flip(image, 1)
				flipped_measurement = measurement * -1.0
				augmented_images.append(flipped_image)
				augmented_measurements.append(flipped_measurement)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(64, 64, 3)))
# model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(6,5,5,input_shape=(64,64,3),activation=('relu')))
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation=('relu')))
model.add(Convolution2D(64,3,3,activation=('relu')))
model.add(Convolution2D(64,3,3,activation=('relu')))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=3, verbose=1)

model.save('model.h5')
