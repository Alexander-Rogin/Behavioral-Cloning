import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	if '\\' in source_path:
		delimiter = '\\'
	else:
		delimiter = '/'
	filename = source_path.split(delimiter)[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	images.append(cv2.flip(image, 1))

	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

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
