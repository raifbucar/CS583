#!/usr/bin/env python
# coding: utf-8

#Upload required libraries

from keras import optimizers
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Conv2D, GlobalAveragePooling1D
from keras.layers import UpSampling2D, Reshape, Dropout
from keras.utils import to_categorical
from keras.regularizers import l1
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import pyarrow.parquet as pq
import numpy as np
import csv
import pickle

#Start by defining functions to read the data in chunks and sepparate the series per phase
def chunk_parquet(filename, chunk_size, init=0):
    a1=np.empty((0,800000))
    a2=np.empty((0,800000))
    a3=np.empty((0,800000))
    a=[a1,a2,a3]
    tab = pq.read_table(filename, columns=[str(x) for x in range(3*init, 3 * (chunk_size+init), 1)])

    for j in range(3):
        a[j]=np.vstack([tab[i+j] for i in range(0, 3 * (chunk_size), 3)])

    for i in range(3):
        a[i]=np.mean(a[i].reshape(-1, 1000,800), axis=2)
    return a

def chunk_target(filename, chunk_size, init=0):
    a1=np.empty((0,1))
    a2=np.empty((0,1))
    a3=np.empty((0,1))
    with open(filename,'r') as file:
        f=csv.reader(file)
        for i in range(3*init+1):
            next(f)
        for i in range(chunk_size):
            a1=np.append(a1,(int(next(f)[3])))
            a2=np.append(a2,(int(next(f)[3])))
            a3=np.append(a3,(int(next(f)[3])))
    a=[a1,a2,a3]
    return a

#Define functions to save the model
def save_binary(athing, filename):
    """
    saves a binary file of the object passed
    :param athing:
    :return:
    """
    try:
        with open('%s.pickle' % (filename, ), 'wb') as handle:
            pickle.dump(athing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except:
        print("Could not save on pickle!!!!")
        return False


def read_binary(filename):
    """
    Reads the binary and returns the object
    :param filename: str The file to be read
    :return: obj The returned object
    """
    with open('%s.pickle'%(filename, ), 'rb') as handle:
        b = pickle.load(handle)
    return b

#Now, start building the dense autoencoder classifier

ser_len=1000

input_AE = Input(shape = (ser_len,), name='input_AE')

norm = BatchNormalization(name='norm_CNN')(input_AE)

dropout = Dropout(0.5, name='encoder_dropout')(norm)

encoder_Den1 = Dense(500, activation='tanh', activity_regularizer=l1(0.0001), name='encoder_CNN1')(dropout)

encoder_Den2 = Dense(200, activation='tanh', activity_regularizer=l1(0.0001), name='encoder_CNN3')(encoder_Den1)

bottleneck = Dense(50, activation='tanh', activity_regularizer=l1(0.0001), name='bottleneck')(encoder_Den2)

decoder_Den1 = Dense(200, activation='tanh', activity_regularizer=l1(0.0001), name='decoder_Den1')(bottleneck)

decoder_Den2 = Dense(500, activation='tanh', activity_regularizer=l1(0.0001), name='decoder_Den2')(decoder_Den1)

dropout2 = Dropout(0.5, name='decoder_dropout')(decoder_Den2)

output_AE = Dense(1000, activity_regularizer=l1(0.0001), name='output_AE')(dropout2)

#Build the classifier for multi-task learning:

classifier_1 = Dense(10, activation='relu', name='classifier_1')(bottleneck)

output_class = Dense(2, activation='softmax', name='output_classifier')(classifier_1)

model=Model(input_AE, [output_AE, output_class])

model.summary()

#Get data, compile and fit the model

chunk_size=2904
filename='train.parquet'
filename2='metadata_train.csv'
model.compile(loss=['mean_squared_error', 'categorical_crossentropy'], loss_weights=[0.3,0.7], optimizer=optimizers.RMSprop(lr=1E-3))

x_tr = np.vstack(chunk_parquet(filename, chunk_size))
y_tr = np.vstack(chunk_target(filename2, chunk_size))

PD_ind=[idx for idx, x in enumerate(y_tr.reshape(3*chunk_size)) if x==1]
non_PD_ind=[idx for idx, x in enumerate(y_tr.reshape(3*chunk_size)) if x==0]
PDs=len(PD_ind)
sw=np.zeros((3*chunk_size,))

for ind in PD_ind:
    sw[ind]=6.
for ind in non_PD_ind:
    sw[ind]=1.

x_tr=x_tr.reshape((3*chunk_size,ser_len,))
y_tr=to_categorical(y_tr.reshape((3*chunk_size,)))

history = model.fit(x_tr, [x_tr, y_tr], batch_size=20, epochs=100, validation_split=0.2, sample_weight=[sw,sw], verbose=2)

#Save the model
save_binary(model, 'trained_model')
save_binary(history, 'evolution_of_the_model')

#Generate the predictions and save them to a .csv
chunk_size=6779
init=2904
filename='test.parquet'

x_ts = np.vstack(chunk_parquet(filename, chunk_size, init))
_, y_pred = model.predict(x_ts)

a=np.array([1-int(round(x[0])) for x in y_pred])

y_ts=a.reshape(3,chunk_size).T.reshape(3*chunk_size,).T

with open('predictions.csv', 'w') as file:
    f=csv.writer(file)
    for i in range(len(y_ts)):
        f.writerow([y_ts[i]])