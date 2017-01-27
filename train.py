# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:12:53 2017

@author: wolf4461
"""

import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from model import buildModel
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc
import scipy.ndimage as ndimage

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

base_path = 'cells/'
data = []
anno = []

def step_decay(epoch):
    step = 16
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)
    
def read_data(base_path):
    imList = os.listdir(base_path)
    for i in range(len(imList)): 
        if 'cell' in imList[i]:
            img1 = misc.imread(os.path.join(base_path,imList[i]))
            data.append(img1)
            
            img2_ = misc.imread(os.path.join(base_path, imList[i][:3] + 'dots.png'))
            img2 = 100.0 * (img2_[:,:,0] > 0)
            img2 = ndimage.gaussian_filter(img2, sigma=(1, 1), order=0)
            anno.append(img2)
    return np.asarray(data, dtype = 'float32'), np.asarray(anno, dtype = 'float32')
    
def train_(base_path):
    data, anno = read_data(base_path)
    anno = np.expand_dims(anno, axis = -1)
    
    mean = np.mean(data)
    std = np.std(data)
    
    data_ = (data - mean) / std
    
    train_data = data_[:150]
    train_anno = anno[:150]

    val_data = data_[150:]
    val_anno = anno[150:]
    
    print('-'*30)
    print('Creating and compiling the fully convolutional regression networks.')
    print('-'*30)    
    # Need to calculate the mean of all the volumes, save it.
   
    model = buildModel(input_dim = (256,256,3))
    model_checkpoint = ModelCheckpoint('cell_counting.hdf5', monitor='loss', save_best_only=True)
    model.summary()
    print('...Fitting model...')
    print('-'*30)
    change_lr = LearningRateScheduler(step_decay)

    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
        zoom_range = 0.3,
        shear_range = 0.,
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True,
        fill_mode = 'constant',
        dim_ordering = 'tf')  # randomly flip images

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_data,
                                     train_anno,
                                     batch_size = 16
                                     ),
                        samples_per_epoch = train_data.shape[0],
                        nb_epoch = 96,
                        callbacks = [model_checkpoint, change_lr],
                       )
    
    model.load_weights('cell_counting.hdf5')
    A = model.predict(val_data)
    mean_diff = (np.sum(A) - np.sum(val_anno)) / (100.0 * val_anno.shape[0] )
    print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
    
if __name__ == '__main__':
    train_(base_path)
