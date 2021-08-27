import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -*- coding: utf-8 -*-
"""
Created on Jan 26, 2018
@author: Alexander Filonenko, Sowmya Kasturi
"""
# Imports
# Keras
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, Conv3D
from keras.layers import MaxPooling3D, BatchNormalization
from keras.utils import np_utils
#from keras import backend as K
#K.set_image_dim_ordering('th')

# OpenCV
import cv2

# Other
import numpy as np


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        print("--- Losses = ({}, {}), [t, v] \n ---Accuracy = ({}%, {}%), [t, v]".format(logs.get('loss'),
                                                                                         logs.get('val_loss'),
                                                                                         logs.get('acc')*100,
                                                                                         logs.get('val_acc')*100))


# Global variables
FRAMES_TO_READ = 20
NUMBER_OF_CLASSES = 2

# image specification
img_rows, img_cols, img_depth = 32, 32, FRAMES_TO_READ  # TODO

# CNN Training parameters
batch_size = 8
nb_classes = NUMBER_OF_CLASSES
nb_epoch = 100

# Training data
X_tr = []  # variable to store entire dataset
X_val = []

num_neg_input_train = 0
num_pos_input_train = 0
num_neg_input_val = 0
num_pos_input_val = 0

# Reading positive samples, training
listing2 = os.listdir('/home/islab/Research/Alexander/Datasets/Falling/Training/color')
for vid2 in listing2:
    vid2 = '/home/islab/Research/Alexander/Datasets/Falling/Training/color/' + vid2
    frames = []
    cap = cv2.VideoCapture(vid2)
    fps = cap.get(5)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    num_pos_input_train = num_pos_input_train + 1
    for k in range(FRAMES_TO_READ):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    input_frames = np.array(frames)

    print(input_frames.shape)
    ipt = np.rollaxis(np.rollaxis(input_frames, 2, 0), 2, 0)
    print(ipt.shape)

    X_tr.append(ipt)

# Reading negative samples, training
listing = os.listdir('/home/islab/Research/Alexander/Datasets/Falling/Training/adl_color')
for vid in listing:
    vid = '/home/islab/Research/Alexander/Datasets/Falling/Training/adl_color/' + vid
    frames = []
    num_neg_input_train = num_neg_input_train + 1
    print("Video file: {0}", vid)
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(FRAMES_TO_READ):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    input_frames = np.array(frames)

    print(input_frames.shape)
    ipt = np.rollaxis(np.rollaxis(input_frames, 2, 0), 2, 0)
    print(ipt.shape)

    X_tr.append(ipt)





x_train = np.zeros((len(X_tr), img_rows, img_cols, FRAMES_TO_READ), dtype='uint8')

print("===> Reshaping training data")
for i in range(x_train.shape[0]):
        print("i={} of {}".format(i, x_train.shape[0]))
        x_train[i, :, :, :] = X_tr[i]

x_train = x_train.astype('float32')
x_train /= 255

# Reading positive samples, validation
listing3 = os.listdir('/home/islab/Research/Alexander/Datasets/Falling/Validation/color')
for vid3 in listing3:
    vid3 = '/home/islab/Research/Alexander/Datasets/Falling/Validation/color/' + vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    num_pos_input_val = num_pos_input_val + 1
    for k in range(FRAMES_TO_READ):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    input_frames = np.array(frames)

    print(input_frames.shape)
    ipt = np.rollaxis(np.rollaxis(input_frames, 2, 0), 2, 0)
    print(ipt.shape)

    X_val.append(ipt)

listing4 = os.listdir('/home/islab/Research/Alexander/Datasets/Falling/Validation/adl_color')

for vid4 in listing4:
    vid4 = '/home/islab/Research/Alexander/Datasets/Falling/Validation/adl_color/' + vid4
    frames = []
    cap = cv2.VideoCapture(vid4)
    fps = cap.get(5)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    num_neg_input_val = num_neg_input_val + 1
    for k in range(FRAMES_TO_READ):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    input_frames = np.array(frames)

    print(input_frames.shape)
    ipt = np.rollaxis(np.rollaxis(input_frames, 2, 0), 2, 0)
    print(ipt.shape)

    X_val.append(ipt)

X_tr_array = np.array(X_tr)  # convert the frames read into array
X_val_array = np.array(X_val)  # convert the frames read into array

num_samples_train = len(X_tr_array)
num_samples_val = len(X_val_array)
print("Number of training and validation samples {} | {}. Ratio = {}".format(num_samples_train,
                                                                             num_samples_val,
                                                                             num_samples_val/num_samples_train))

# Assign Label to each class
label_train = np.ones((num_samples_train,), dtype=int)
label_val = np.ones((num_samples_val,), dtype=int)

"""
label_train[0:100]= 0
label_train[100:199] = 1
label_train[199:299] = 2
label_train[299:399] = 3
label_train[399:499]= 4
label_train[499:] = 5
"""
label_train[0:num_pos_input_train] = 0
label_train[num_pos_input_train:(num_pos_input_train+num_neg_input_train)] = 1

label_val[0:num_pos_input_val] = 0
label_val[num_pos_input_val:(num_pos_input_val+num_neg_input_val)] = 1

train_data = [X_tr_array, label_train]
val_data = [X_val_array, label_val]

(X_train, y_train) = (train_data[0], train_data[1])
(X_val, y_val) = (val_data[0], val_data[1])

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth, 1)
X_val= X_val.reshape(X_val.shape[0], img_rows, img_cols, img_depth, 1)

print('X_Train shape: ', X_train.shape)

#train_set = np.zeros((num_samples_train, 1, img_rows, img_cols, img_depth))
#val_set = np.zeros((num_samples_val, 1, img_rows, img_cols, img_depth))

train_set = np.asarray(X_train)
val_set = np.asarray(X_val)
#for h in range(num_samples_train):
#    train_set[h][0][:][:][:] = X_train[h, :, :, :]

#for h in range(num_samples_val):
#    val_set[h][0][:][:][:] = X_val[h, :, :, :]

patch_size = FRAMES_TO_READ  # img_depth or number of frames used for each video

print(train_set.shape, ' train samples')


print("===> Training parameters. Batch = {}, classes = {}, epochs = {}".format(batch_size, nb_classes, nb_epoch))


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

# number of convolutional filters to use at each layer
nb_filters = [  32,   # 1st conv layer
                32,   # 2nd
                32
             ]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3, 3]



# Pre-processing
train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /= np.max(train_set)

val_set = val_set.astype('float32')
val_set -= np.mean(val_set)
val_set /= np.max(val_set)



# autosave best Model
best_model_file = ('best_model_training.h5')
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
history = LossHistory()
csv_logger = CSVLogger('training.csv')
tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Define model
#input_shape = (1, img_rows, img_cols, img_depth)
input_shape = (img_rows, img_cols, img_depth, 1)
model = Sequential()
model.add(Conv3D(nb_filters[0],
                 (nb_conv[0], nb_conv[0], FRAMES_TO_READ),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))

#model.add(MaxPooling3D((nb_pool[0], nb_pool[0], 1)))

# Mini-netwok
model.add(Conv3D(nb_filters[1], kernel_size=(nb_conv[1], nb_conv[1], nb_conv[1]),
                 activation='relu',
                 padding='same'))

model.add(MaxPooling3D((nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Conv3D(nb_filters[2], kernel_size=(nb_conv[2], nb_conv[2], nb_conv[2]), activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'mse'])

"""
#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth, 1)
#X_val= X_val.reshape(X_val.shape[0], img_rows, img_cols, img_depth, 1)
#input_shape = (img_rows, img_cols, img_depth, 1)
input_shape = (1, img_rows, img_cols, img_depth)

model = Sequential()

print(nb_filters[0], 'filters')
print('input shape', img_rows, 'rows', img_cols, 'cols', patch_size, 'patchsize')

model.add(Convolution3D(
    nb_filters[0],
    kernel_dim1=nb_conv[0],  # depth
    kernel_dim2=nb_conv[0],  # rows
    kernel_dim3=nb_conv[0],  # cols
    input_shape=input_shape,
    activation='relu'
))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])
"""


model.summary()
# Split the data

#X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

# Train the model

hist = model.fit(train_set, Y_train, validation_data=(val_set, Y_val),
                 batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1,
                 callbacks=[best_model, history, csv_logger, tbCallBack])

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)
print("___________________________________________")
print("Losses")
print(history.losses)
print("---------------------")
print("acc")
print(history.accs)
print("---------------------")
print("val_loss")
print(history.val_losses)
print("---------------------")
print("val_acc")
print(history.val_accs)
print("___________________________________________")
model.save("Model1_ir.h5")


print("Finished fitting model")
score = model.evaluate(val_set, Y_val, verbose=1)
print('Test loss:', score[0])
print('Test mean squared error:', score[2])
print('Test accuracy:', score[1])