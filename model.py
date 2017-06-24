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

	#randomize brightness
	img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	randbrightness = .2 + np.random.uniform()
	img[:,:,2] = img[:,:,2] * randbrightness
	img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

	# normalize image
	image = (img / 127.5) - 1.0

	return image

# fit_generator function
BATCH_SIZE = 32
def generator(samples, BATCH_SIZE):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, BATCH_SIZE):
			batch_samples = samples[offset:offset+BATCH_SIZE]

			# get images and steering angles
			images = []
			measurements = []
			angle_range = 0.5
			delete_rate = 0.7
			correction = 0.2
			for batch_sample in batch_samples:
				## set filter for steering angle measurement
				measurement = float(batch_sample[3])
				# if the steering angle is not within -0.5 to 0.0 to 0.5
				# include all 3 images in the batch sample
				if (abs(measurement) > angle_range):
					for j in range(3):
						# open/read image files
						source_path = batch_sample[j].split('\\')[-1]
						filename = source_path.split('/')[-1]
						local_path = './data/IMG/' + filename
						image = cv2.imread(local_path)
						image = process_image(image)
						images.append(image)
					measurements.append(measurement)
					measurements.append(measurement + correction)
					measurements.append(measurement - correction)
				else:
					# if not, the delete rate will randomly determine which
					# image/images in the batch sample will be included
					for j in range(3):
						if (np.random.random() > delete_rate):
							# open/read image files
							source_path = batch_sample[j].split('\\')[-1]
							filename = source_path.split('/')[-1]
							local_path = './data/IMG/' + filename
							image = cv2.imread(local_path)
							image = process_image(image)
							images.append(image)
							if (j == 0): measurements.append(measurement)
							if (j == 1): measurements.append(measurement + correction)
							if (j == 2): measurements.append(measurement - correction)

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
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Convolution2D(24,5,5,input_shape=(64,64,3),subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0), ModelCheckpoint('model.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only = True, verbose = 0),]

samples_per_epoch = len(train_samples) // BATCH_SIZE * BATCH_SIZE * 6

history_object = model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch, validation_data=validation_generator, nb_val_samples=len(validation_samples)//BATCH_SIZE*BATCH_SIZE, nb_epoch=50, verbose=1, callbacks=callbacks)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.subplot(111)
plt.plot(history_object.history['loss'], 'ro-')
plt.plot(history_object.history['val_loss'], 'bo-')
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show(block=True)
plt.savefig('test.png')
