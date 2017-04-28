import csv
import cv2
import numpy as np
import random

def read_image(imagePath):
	if '\\' in imagePath:
		delimiter = '\\'
	else:
		delimiter = '/'
	filename = imagePath.split(delimiter)[-1]
	current_path = './data/IMG/' + filename
	return cv2.imread(current_path)

def process_image(images, measurements, imagePath, measurement, useAugmented):
	image = read_image(imagePath)
	images.append(image)
	measurements.append(measurement)

	if useAugmented: #and random.randint(0, 1) == 1:
		images.append(cv2.flip(image, 1))
		measurements.append(-measurement)
#	else:
#		images.append(image)
#		measurements.append(measurement)

def load_data(useAugmented=False, useMultipleCameras=False):
	images = []
	measurements = []
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)
		for line in reader:
			imagePath = line[0]
			measurement = float(line[3])
#			if abs(measurement) <= 0.05 and random.randint(0, 3) != 0:
#				continue
			process_image(images, measurements, imagePath, measurement, useAugmented)

			if useMultipleCameras:
				correction = 0.1

				leftImagePath = line[1]
				measurementLeft = measurement + correction
				process_image(images, measurements, leftImagePath, measurementLeft, useAugmented)

				rightImagePath = line[1]
				measurementRight = measurement - correction
				process_image(images, measurements, rightImagePath, measurementRight, useAugmented)

	return np.array(images),np.array(measurements)



X_train, y_train = load_data(useAugmented=True, useMultipleCameras=False)

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout

shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), strides=(2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
