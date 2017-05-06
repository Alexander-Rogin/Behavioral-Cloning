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

    if useAugmented and random.randint(0, 1) == 1:
        images.append(cv2.flip(image, 1))
        measurements.append(-measurement)
    else:
        images.append(image)
        measurements.append(measurement)

def load_data(useAugmented=False, useMultipleCameras=False):
    images = []
    measurements = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            imagePath = line[0]
            measurement = float(line[3])
            # if abs(measurement) <= 0.01:
            #   continue
            process_image(images, measurements, imagePath, measurement, useAugmented)

            if useMultipleCameras:
                correction = 0.35

                leftImagePath = line[1]
                measurementLeft = measurement + correction
                process_image(images, measurements, leftImagePath, measurementLeft, useAugmented)

                rightImagePath = line[1]
                measurementRight = measurement - correction
                process_image(images, measurements, rightImagePath, measurementRight, useAugmented)

    return np.array(images),np.array(measurements)



samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
import sklearn
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def addImage(images, angles, image, angle, useAugmented):
    images.append(image)
    angles.append(angle)
    if useAugmented:
        images.append(cv2.flip(image, 1))
        angles.append(-angle)

def generator(samples, batch_size=32, useAugmented=False, useMultipleCameras=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if abs(center_angle) <= 0.05:
                    continue
                addImage(images, angles, center_image, center_angle, useAugmented)
                if useMultipleCameras:
                    correction = 0.2
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    addImage(images, angles, left_image, center_angle + correction, useAugmented)

                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    addImage(images, angles, right_image, center_angle - correction, useAugmented)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64, useAugmented=True, useMultipleCameras=False)
validation_generator = generator(validation_samples, batch_size=64, useAugmented=True, useMultipleCameras=False)


from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout

shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape, output_shape=shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(48, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)

model.save('model.h5')