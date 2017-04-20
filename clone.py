import csv
import cv2
import numpy as np

def read_image(source_path):
	if '\\' in source_path:
		delimiter = '\\'
	else:
		delimiter = '/'
	filename = source_path.split(delimiter)[-1]
	current_path = './data/IMG/' + filename
	return cv2.imread(current_path)

def load_data(useAugmented=False, useMultipleCameras=False):
	images = []
	measurements = []
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			source_path = line[0]
			image = read_image(source_path)
			images.append(image)
			if useAugmented:
				images.append(cv2.flip(image, 1))
			if useMultipleCameras:
				img_left = cv2

			measurement = float(line[3])
			measurements.append(measurement)
			if useAugmented:
				measurements.append(-measurement)


	return np.array(images),np.array(measurements)

X_train, y_train = load_data(useAugmented=True, useMultipleCameras=True)

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout

shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(6, (5, 5), input_shape=shape, activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(Dropout(0.5))

model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=15)

model.save('model.h5')
