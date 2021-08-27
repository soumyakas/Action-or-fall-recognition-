import keras
import numpy as np
# OpenCV
#import cv2
import os
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import skimage

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dict_IDs, num_frames, batch_size=32, dim=(32,32), n_channels=1,
                 n_classes=2, shuffle=True, is_cnnrnn_format=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dict_IDs = dict_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.num_frames = num_frames

        self.dict_IDs = dict_IDs
        self.is_cnnrnn_format = is_cnnrnn_format

        self.list_IDs = []

        for key, value in dict_IDs.items():
            self.list_IDs.append(key)

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp = []
        for _, k in enumerate(indexes):
            #print ("k = {}".format(k))
            temp = self.list_IDs[k]
            #print(temp)

            list_IDs_temp.append(self.list_IDs[k])




        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        X = np.asarray(X)
        #print('before ', X.shape)

        #print("Before reshaping", X.shape)
        if self.is_cnnrnn_format:
            X = X.reshape(X.shape[0], self.num_frames, self.dim[0], self.dim[1], 1)
        else:
            X = X.reshape(X.shape[0], self.dim[0], self.dim[1], self.num_frames, 1)
        #print("After reshaping", X.shape)
        #print('after ', X.shape)


        # Pre-processing
        X = X.astype('float32')
        X -= np.mean(X)
        X /= np.max(X)

        #print("Data shape {}".format(X.shape))

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        X = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            frames = self.read_frames(ID, 'frame ({}).png', self.num_frames)

            input_frames = np.array(frames)

            #print(input_frames.shape)
            if not self.is_cnnrnn_format:
                #print('Original 3DCNN format')
                ipt = np.rollaxis(np.rollaxis(input_frames, 2, 0), 2, 0)
                X.append(ipt)
            else:
                X.append(input_frames)
                #print('CNN+RNN format', input_frames.shape)
            #print('ipt ', ipt.shape)
            #X[i,] = ipt



            # Store class
            y[i] = self.dict_IDs[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def read_frames(self, folder, template, number):
        frames = []
        for i in range(number):
            name = os.path.join(folder, template.format(i + 1))
            frame = io.imread(name, as_grey=True)
            #print(frame.shape)
            #image_resized = skimage.transform.resize(frame, (self.dim[0], self.dim[1]))
            #frame_pil.thumbnail((self.dim[0], self.dim[1]), Image.ANTIALIAS, )
            #frame = np.array(frame_pil)
            #print(image_resized.shape)
            #gray = self.rgb2gray(frame)
            #frame = cv2.resize(frame, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_CUBIC)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(gray.shape)
            frames.append(frame)
        return frames

    def rgb2gray(self, rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
