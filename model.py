import csv
import cv2
import numpy as np

path = ['./data_20170614_1600_1/', './data_20170616_0015_2/', '/data_20170616_1200_3/']
lines = []
images = []
measurements = []
for i in range(1):
	token = path[i] + 'driving_log.csv'
	with open(token, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
		
	for line in lines:
		for j in range(3):
			source_path = line[j].split('\\')[-1]
			filename = source_path.split('/')[-1]
			local_path = path[i] + 'IMG/' + filename
			image = cv2.imread(local_path)
			images.append(image)
		correction = 0.25
		measurement = float(line[3])
		measurements.append(measurement)
		measurements.append(measurement + correction)
		measurements.append(measurement - correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = measurement * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

