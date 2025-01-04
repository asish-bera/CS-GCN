from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from spektral.layers import  GlobalAttentionPool,SortPool,TopKPool, GlobalSumPool, GlobalAttnSumPool, TAGConv, APPNPConv,GlobalAvgPool, AGNNConv
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Dropout, Flatten
from spektral.layers.convolutional.conv import Conv
from tensorflow.keras import backend as K
from Attention import Attention
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, GlobalSumPool
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import sys
import os
from tensorflow.keras.models import Sequential
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, GlobalSumPool
from Attention import Attention
from keras_self_attention import SeqWeightedAttention as Attention
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import top_k_categorical_accuracy
from DG_15RoIs import DirectoryDataGenerator
from AG_Net_RoiPoolingConvTF2 import RoiPoolingConv
from layer_normalization import LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from SelfAttention import SelfAttention
import numpy as np
from SpatialPooling import SpatialPyramidPooling
from custom_validate_callback import CustomCallback
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_input

import seaborn as sn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#~~~~~~~~~~~~~~Editable parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#change the base model (e.g. InceptionV3, MobileNetV2, Xception, ResNet50)
from tensorflow.keras.applications.resnet50 import ResNet50  #as KerasModel
#all keras models use the same preprocessing function so this model name is not required to match the above model name (though it can be changed for readability).)
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_input

image_size = (224,224) #image resolution in pixels

#~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#The following should not need to be edited but can be if required. The main reason to do so would be if a dataset is labelled as train & test rather than train & val.
input_tensor = Input(shape=(image_size[0], image_size[1], 3))
train_data_dir = '{}/train/'.format(dataset_dir)
val_data_dir = '{}/val'.format(dataset_dir)
nb_classes = len(os.listdir(train_data_dir))
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)])
verbose = True
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']


def crop(dimension, start, end): #https://github.com/keras-team/keras/issues/890
    #Use this layer for a model that has individual roi bounding box
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)

def squeezefunc(x):
    return K.squeeze(x, axis=1)

'''This is to convert stacked tensor to sequence for LSTM'''
def stackfunc(x):
    return K.stack(x, axis=1)

ROIS_resolution =48
pool_size=4
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']
learning_rate=0.005
lr=0.005
num_rois = 15

fig_dimensions = (6, 6) 
show_legend = True

########## Proposed CNN stem ######################

def make_model(input_shape, num_classes):
    input1 = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(input1)
    x = layers.SeparableConv2D(128, 3, strides=2, padding="same")(input1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation.gelu(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 768, 1024]:
        x = tf.keras.activations.gelu(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(0.25)(x)

        # Project residual
        residual = layers.SeparableConv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x) 

    xy = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    yx = layers.AveragePooling2D(3, strides=2, padding="same")(x)    
    x = layers.add([xy, yx])  
    x = layers.SeparableConv2D(2048, 3, padding="same")(x)
           
    x = layers.BatchNormalization()(x)
    base_x = tf.keras.activations.gelu(x)

    print("final_feature.shape" , base_x.shape)
    ROIS_resolution =48
    full_img = layers.Lambda(lambda x: tf.image.resize(x,size=(ROIS_resolution, ROIS_resolution)), name='Lambda_img_1')(base_x) #Use bilinear upsampling (default tensorflow image resize) to a reasonable size

    print("full_img.shape" , full_img.shape)
    rois = layers.Input(shape=(num_rois, 4), name='RoI')
    print(rois)
    roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois)([full_img, rois])
    base_channels=2048
    feat_dim=4*4*2048
    jc = []

    for j in range(num_rois):
    	roi_crop = crop(1, j, j+1)(roi_pool)
    	lname = 'roi_lambda_48p'+str(j)
    	x = layers.Lambda(squeezefunc, name=lname)(roi_crop)    
    	x = layers.Reshape((feat_dim,))(x)
    	jc.append(x)      

    x = layers.Reshape((feat_dim,))(base_x)
    jc.append(x)

    jc = layers.Lambda(stackfunc, name='lambda_stack')(jc)
    jcvs=tf.keras.layers.Dropout(0.2) (jc)
    x1 = layers.TimeDistributed(layers.Reshape((pool_size,pool_size, base_channels)))(jc)
    print(x1)

    x1 = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time4'))(x1)
    N=num_rois+1
    A=np.ones((N,N), dtype='int')
    fltr = GCNConv.preprocess(A).astype('f4')
    A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr), name='AdjacencyMatrix1')
    
    N4=4
    B=np.ones((N4,N4), dtype='int')
    fltr4 = GCNConv.preprocess(B).astype('f4')
    B_in = Input(tensor=sp_matrix_to_sp_tensor(fltr4), name='AdjacencyMatrix2')
    gc1=GCNConv(channels=1024*2, activation='relu', dropout_rate=0.30) ([x1, A_in])
    print("GNNgc1.shape:", gc1.shape)

    p1 = layers.GlobalAveragePooling1D()(gc1)
    p1= layers.Reshape((1,2048))(p1)
    p2 = layers.GlobalMaxPooling1D()(gc1)
    p2= layers.Reshape((1,2048))(p2)

    p5 = layers.concatenate([p1, p2], axis=1)  # Add back residual
    p6= layers.Reshape((N4, 1024))(p5)
    gc3=GCNConv(channels=1024, activation='relu', dropout_rate=0.30) ([p6, B_in])
    print("GNNgc3.shape:", gc3.shape) 
    p3 = layers.GlobalAveragePooling1D()(gc3)
   
    x4 = layers.Dropout(0.2)(p3)
    bn=BatchNormalization(name='bN')(x4)
    op= layers.Dense(nb_classes, activation='softmax')(bn)
    return tf.keras.Model(inputs=[input1, rois, A_in, B_in], outputs=op)

model = make_model(input_shape=(224,224,3), num_classes=nb_classes)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()



###################### full Model ####################



