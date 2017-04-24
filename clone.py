import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

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

	if useAugmented:
		images.append(cv2.flip(image, 1))
		measurements.append(-measurement)

def load_data(useAugmented=False, useMultipleCameras=False):
	images = []
	measurements = []
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		firstLine = True
		for line in reader:
			if firstLine:
				firstLine = False
				continue
			imagePath = line[0]
			measurement = float(line[3])
			process_image(images, measurements, imagePath, measurement, useAugmented)

			if useMultipleCameras:
				correction = 0.2

				leftImagePath = line[1]
				measurementLeft = measurement + correction
				process_image(images, measurements, leftImagePath, measurementLeft, useAugmented)

				rightImagePath = line[1]
				measurementRight = measurement - correction
				process_image(images, measurements, rightImagePath, measurementRight, useAugmented)

	return np.array(images),np.array(measurements)


samples = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32, useAugmented=False, useMultipleCameras=False):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				name = './IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)

				if useAugmented:
					images.append(cv2.flip(center_image, 1))
					angles.append(-center_angle)

				if useMultipleCameras:
					correction = 0.2

					name = './IMG/'+batch_sample[1].split('/')[-1]
					left_image = cv2.imread(name)
					left_angle = center_angle + correction

					name = './IMG/'+batch_sample[2].split('/')[-1]
					right_image = cv2.imread(name)
					right_angle = center_angle - correction

					images.append(left_image)
					angles.append(left_angle)
					images.append(right_image)
					angles.append(right_angle)

					if useAugmented:
						images.append(cv2.flip(left_image, 1))
						angles.append(-left_angle)
						images.append(cv2.flip(right_image, 1))
						angles.append(-right_angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, useAugmented=True, useMultipleCameras=True)
validation_generator = generator(validation_samples, batch_size=32, useAugmented=True, useMultipleCameras=True)



#X_train, y_train = load_data(useAugmented=True, useMultipleCameras=True)

from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout

#shape = (160, 320, 3)
ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
#model.add(Cropping2D(cropping=((70, 25), (0, 0))))

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
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)
model.fit_generator(train_generator, samples_per_epoch= / len(train_samples), validation_data=validation_generator, / nb_val_samples=len(validation_samples), nb_epoch=1)

#model.save('model.h5')
