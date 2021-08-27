import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# -*- coding: utf-8 -*-
"""
Created on Jan 26, 2018
Updated on Apr 22, 2018
@author: Alexander Filonenko, Sowmya Kasturi
"""
# Imports
from os import walk
# Keras
print("STARTING")
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers import MaxPooling3D
from keras.datasets import mnist
from keras.models import Sequential
# from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import advanced_activations
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, Adadelta, Adam, SGD
from keras.layers.convolutional import Convolution2D, AtrousConvolution2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model
from keras.utils import multi_gpu_model


print("LOADED STANDARD LIBRARIES")
from Datagenerator import DataGenerator
print("LOADED DataGenerator")
import folder_list
print("LOADED folder_list")



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
NUMBER_OF_CLASSES = 14
NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 2
NUMBER_OF_CHANNELS = 1


# Parameters
params = {'dim': (128, 128),
          'num_frames': 20,
          'batch_size': BATCH_SIZE,
          'n_classes': NUMBER_OF_CLASSES,
          'n_channels': NUMBER_OF_CHANNELS,
          'shuffle': True,
          'is_cnnrnn_format': True}

nb_epochs = NUMBER_OF_EPOCHS
batch_size = 2
split = 0.1
nb_classes = NUMBER_OF_CLASSES
nb_channels = params['n_channels']
inputFrameSize = params['dim'][0]
conv1_fm = 64
conv1_kernel_size = 3

# Datasets
# Load the list of folders in the training dataset folder
print("LOADING DATASET DATA")
train_dict, val_dict = folder_list.split_train_val("/home/islab/Research/Sowmya/Datasets/frames_resized_128_cropped/islab frames/Color/", ratio=0.8)


# Generators
print("INITIALIZING DATA GENERATOR")
training_generator = DataGenerator(train_dict, **params)
validation_generator = DataGenerator(val_dict, **params)


print("BUILDING A MODEL")
# number of convolutional filters to use at each layer
nb_filters = [  64,   # 1st conv layer
                256,   # 2nd
                512
             ]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3, 3]




# autosave best Model
best_model_file = ('best_model_training.h5')
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
history = LossHistory()
csv_logger = CSVLogger('training.csv')
#tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Define model
#input_shape = (1, img_rows, img_cols, img_depth)
print("Building model")
model = Sequential()
model.add(TimeDistributed(Convolution2D(conv1_fm, conv1_kernel_size, conv1_kernel_size, border_mode='valid'), \
                          input_shape=(params['num_frames'], inputFrameSize, inputFrameSize, nb_channels)))
model.add(Activation('relu'))
#model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same')))
##model.add(Activation('relu'))
#model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same')))
#model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

model.add(TimeDistributed(AtrousConvolution2D(64, 3, 3, atrous_rate=(1, 2), border_mode='same', activation='relu')))
model.add(TimeDistributed(AtrousConvolution2D(64, 3, 3, atrous_rate=(2, 1), border_mode='same', activation='relu')))
model.add(TimeDistributed(AtrousConvolution2D(64, 3, 3, atrous_rate=(2, 2), border_mode='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

model.add(TimeDistributed(Dense(64)))
model.add(TimeDistributed(Flatten()))
model.add(advanced_activations.ELU())

model.add(GRU(output_dim=64, return_sequences=False))
model.add(Dropout(.2))

model.add(Dense(nb_classes))
model.add(Activation('softmax', name='softmax_activation'))

#model = multi_gpu_model(model, gpus=2)

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
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    nb_epoch=NUMBER_OF_EPOCHS,
                    callbacks=[best_model, history, csv_logger])


#hist = model.fit(train_set, Y_train, validation_data=(val_set, Y_val),
#                 batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1,
#                 callbacks=[best_model, history, csv_logger, tbCallBack])

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
model.save("cnn_rnn_color_june22.h5")


print("Finished fitting model")
#score = model.evaluate(val_set, Y_val, verbose=1)
#print('Test loss:', score[0])
#print('Test mean squared error:', score[2])
#print('Test accuracy:', score[1])