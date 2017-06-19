import csv
import cv2
import numpy as np
import sklearn

# open/read data file
samples = []
with open('./data_combo123/main_driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
	print('Number of lines on driving log: ', len(samples))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

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
					source_path = batch_sample[j].split('\\')[-1]
					filename = source_path.split('/')[-1]
					local_path = './data_combo123/IMG/' + filename
					image = cv2.imread(local_path)
					images.append(image)
				correction = 0.25
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
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#ch, row, col = 3, 80, 320  # Trimmed image format

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(6,5,5,activation=('relu')))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(16,5,5,activation=('relu')))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')

