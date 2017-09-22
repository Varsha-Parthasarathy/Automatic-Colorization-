#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:08:14 2017

@author: varshaparthasarathi
"""
import tensorflow as tf
#import keras
import numpy as np
from keras import optimizers
from keras import losses
from keras.models import Sequential,Model
from keras.models import model_from_json
import h5py
from keras.layers import *
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, UpSampling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import argparse
import itertools
from os import listdir
from skimage import io, color

size = 32

def get_LAB_from_RGB1(color_file):
    
    rgb = io.imread(color_file)
    Lab = color.rgb2lab(rgb)
    L = Lab[:,:,0]
    a = Lab[:,:,1]
    b = Lab[:,:,2]
    return L,a,b



        

#%%

#Model starts here
inputs = Input(shape=(size,size,3)) 
#repeat = RepeatVector(3)(inputs)
B0 = BatchNormalization()(inputs)

z1 = ZeroPadding2D((1,1),input_shape=(1,1,size,size))(inputs)
conv1 = Convolution2D(64,(3,3), activation='relu')(z1)
B1 = BatchNormalization()(conv1)
mpool1 = MaxPooling2D(pool_size=(2,2))(conv1)

z2 = ZeroPadding2D((1,1))(mpool1)
conv2 = Convolution2D(128, (3, 3), activation='relu')(z2)
B2 = BatchNormalization()(conv2)
mpool2 = MaxPooling2D(pool_size=(2,2))(conv2)

z3 = ZeroPadding2D((1,1))(mpool2)
conv3 = Convolution2D(256, (3, 3), activation='relu')(z3)
B3 = BatchNormalization()(conv3)
mpool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

z4 = ZeroPadding2D((1,1))(mpool3)
conv4 = Convolution2D(512, (3, 3), activation='relu')(z4)
B4 = BatchNormalization()(conv4)

conv5 = Convolution2D(256, (1, 1), activation='relu')(B4)
U1 = UpSampling2D((2,2))(conv5)

BU1 = add([B3, U1])
conv6 = Convolution2D(128, (3,3), activation='relu')(BU1)
conv6 = ZeroPadding2D((1,1))(conv6)
U2 = UpSampling2D((2,2))(conv6)

BU2 = add([B2, U2])
conv7 = Convolution2D(64, (3,3), activation='relu')(BU2)
conv7 = ZeroPadding2D((1,1))(conv7)
U3 = UpSampling2D((2,2))(conv7)

BU3 = add([B1, U3])
conv8 = Convolution2D(3, (3,3), activation='relu')(BU3)
conv8 = ZeroPadding2D((1,1))(conv8)

BU4 = add([B0, conv8])
conv9 = Convolution2D(3, (3,3), activation='relu')(BU4)
conv9 = ZeroPadding2D((1,1))(conv9)

predictions = Convolution2D(2, (3,3), activation='sigmoid', padding='same')(conv9)



model = Model(inputs=inputs, outputs=predictions)
#print model.summary()

##load json and create model
#json_file = open("modelcpu.json", 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
##load weights into new model
#model.load_weights("modelcpu.h5")
#print("Loaded model from disk")

#%%
sgd = optimizers.SGD(lr=0.01, momentum = 0.5, clipnorm=1.)
#%%
# CHANGE LOSS to crossentropy
model.compile(sgd, loss = losses.mean_squared_error)
#%%
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


num_train = 1000
training_files = "./train/"
A = listdir(training_files)
ll = np.zeros((num_train,size,size,3))
aa = np.zeros((size,size))
bb = np.zeros((size,size))
ab = np.zeros((num_train,size,size,2))
LL = []
AA = []
BB = []
YY = []

x = np.arange(50000)
y = np.arange(50000)
for jj in frange(0,50000,num_train):
	for ii in frange(0,num_train,1):

	    #[L1,a1,b1] = get_LAB_from_RGB(training_files + A[ii]) 
	    [L1,a1,b1] = get_LAB_from_RGB1(training_files + A[jj+ii]) 
	    #ll[ii,:,:,0:3] = L1,L1,L1
	    ab[ii,:,:,0] = (a1.astype(np.float32)+127)/255
	    ab[ii,:,:,1] = (b1.astype(np.float32)+127)/255 
	    ll[ii,:,:,0] = L1.astype(np.float32)/100
	    ll[ii,:,:,1] = L1.astype(np.float32)/100
	    ll[ii,:,:,2] = L1.astype(np.float32)/100

	#Change here
	X_train = ll
	#Y_train = YY
	Y_train = ab
	x_train = X_train
	y_train = Y_train
	#%%
	model.fit( x= x_train, y=y_train, batch_size=10, epochs=50, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
	print model

	# serialize model to JSON
	model_json = model.to_json()
	with open("modelcifar"+str(jj)+".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("modelcifar"+str(jj)+".h5")
	print("Saved model into h5 file")
	
	# later...

	# load json and create model
	#json_file = open("model"+str(jj)+".json", 'r')
	#loaded_model_json = json_file.read()
	#json_file.close()
	#loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	#loaded_model.load_weights("model"+str(jj)+".h5")
	#print("Loaded model from disk")
	#print model


#%%
#model.evaluate(x = X_train[1000:1001,:,:,:], y=Y_train[1000:1001,:,:,:], batch_size=1, verbose=1, sample_weight=None)

#%%
test_num = 50000
[L1,a1,b1] = get_LAB_from_RGB1(training_files + A[test_num]) 
ll[0,:,:,0] = L1.astype(np.float32)/100
ll[0,:,:,1] = L1.astype(np.float32)/100
ll[0,:,:,2] = L1.astype(np.float32)/100
X_test=ll
out = model.predict( x = X_test[0:1,:,:,:], batch_size=1, verbose=1)

#%%
L = (X_test[0,:,:,:]*100)
a = (out[0,:,:,0]/np.max((np.max(out[0,:,:,0]),1))*255)-127
b = (out[0,:,:,1]/np.max((np.max(out[0,:,:,1]),1))*255)-127

#%%
LAB = np.zeros((size,size,3))
LAB[0:size,0:size,0] = L[:,:,0]
LAB[0:size,0:size,1] = a
LAB[0:size,0:size,2] = b
   
#output_img = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
output_img = color.lab2rgb(LAB)
#plt.imshow('nsadv',output_img)
file_name = "out" + str(test_num) + "1.jpg"
io.imsave(file_name,output_img)
