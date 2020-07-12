import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import math
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D,Convolution2D,Activation
import keras
import random
from keras.backend import image_data_format
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

retrain = 0
# validation_steps_per_epoch = 0

#collect samples from the driving_log csv file
def collectsamples(csvfile):    
    samples = []
    with open(csvfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


#generator function runs continously and feed training data to the keras fit_genertor
def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    path = '../lastsample/IMG/'
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #center image
                _,name = os.path.split(batch_sample[0])
                center_image = cv2.imread(path + name)
                center_angle = float(batch_sample[3])
                flip_H_center_image = cv2.flip(center_image, 1)
                flip_center_angle = center_angle * (-1)
                
                #left image
                _,name = os.path.split(batch_sample[1])
                left_image = cv2.imread(path + name)
                left_angle = float(batch_sample[3]) + 0.2
                flip_H_left_image = cv2.flip(left_image, 1)
                flip_left_angle = left_angle * (-1)                
                #right image
                _,name = os.path.split(batch_sample[2])
                right_image = cv2.imread(path + name)
                right_angle = float(batch_sample[3]) - 0.3
                flip_H_right_image = cv2.flip(right_image, 1)
                flip_right_angle = right_angle * (-1)   
                
                img_lst = [center_image,flip_H_center_image,left_image,flip_H_left_image,right_image,flip_H_right_image]
                for imag in img_lst:
#                     grey = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
#                     img = grey[..., None]
#                     print(img.shape)
                    images.append(imag)
                
                angles.append(center_angle)
                angles.append(flip_center_angle)
                angles.append(left_angle)
                angles.append(flip_left_angle)
                angles.append(right_angle)
                angles.append(flip_right_angle)
                
            images,angles = shuffle(images,angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

 #Function to visualize the original and augmented data           
def AngleDistribution(samples):
    
    num_samples = len(samples)
    angles = []
    for offset in range(0, num_samples, num_samples):
        for batch_sample in samples:
            center = float(batch_sample[3])   
            angles.append(center)
            angles.append(-center)
            angles.append(center+0.2)
            angles.append(-(center+0.2))
            angles.append(center-0.22)
            angles.append(-(center-0.22))
    return angles

#Function to visualize the original  data

def AngleDistributionOriginal(samples):
    
    num_samples = len(samples)
    angles = []
    for offset in range(0, num_samples, num_samples):
        for batch_sample in samples:
            center = float(batch_sample[3])   
            angles.append(center)
            angles.append(center+0.2)
            angles.append(center-0.22)
    return angles

if __name__ == "__main__":
    # samples collected from the traaining log
    samples = []
    path = '../lastsample/driving_log.csv'
    samples = collectsamples(path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)    

    #data set from the simulator
    anglesori = AngleDistributionOriginal(samples)
    countsori, binsori = np.histogram(anglesori,bins=10)    
    print("Total data points from the training data is",np.sum(countsori))
#     plt.hist(binsori[:-1], binsori, weights=countsori,width=0.1)
#     plt.xlabel("steering angle")
#     plt.ylabel("count")
#     plt.title("Data from the simulator")
#     plt.show()

    #data set from the simulator + Augmentation
    anglesAug = AngleDistribution(samples)
    countstotal, binstotal = np.histogram(anglesAug,bins=10)    
    print("Total data points from the training data + Augment data is",np.sum(countstotal))
#     plt.hist(binstotal[:-1], binstotal, weights=countstotal,width=0.1)
#     plt.xlabel("steering angle")
#     plt.ylabel("count")
#     plt.title("Data from the simulator + Augmented")
#     plt.show()
    
    
    # Set our batch size
    batch_size=32
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples,batch_size=batch_size)
    validation_generator = generator(validation_samples,batch_size=batch_size)
    
    

    #retrain the same model
    filename = "model.h5"
    #retrain epoch
    epochs  = 1   
    if retrain == 1:
        model = load_model(filename)
    else:
        #train epoch
        epochs = 5
        activation_type = 'relu'
        
        #seqeunctial model     
        model = Sequential()

        #cropping and prepoccesing
        model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: (x / 255) - .5))
        #layer 1
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type))

        #layer2
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type))

        #layer3
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type))

        #layer4
        model.add(Conv2D(64, (3, 3), activation=activation_type))

        #layer5
        model.add(Conv2D(64, (3, 3), activation=activation_type))

        
        model.add(Flatten())
        
        model.add(Dropout(0.5)) # ab: attempt to avoid overfitting
        
        model.add(Dense(1162, activation=activation_type))
        model.add(Dense(100, activation=activation_type))
        model.add(Dense(50, activation=activation_type))
        model.add(Dense(10, activation=activation_type))
        model.add(Dense(1,activation='linear'))
        model.summary()
        
        model.compile(loss='mse', optimizer='adam')
    
    #call back function to save the model only when the training loss decrease
    checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]    
    
    history = model.fit_generator(train_generator, \
                                  steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
                                  validation_data=validation_generator, \
                                  validation_steps=math.ceil(len(validation_samples)/batch_size), \
                                  epochs=epochs, verbose=1,callbacks=callbacks_list)
