
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from spektral.layers import  GlobalAttentionPool,SortPool,TopKPool, GlobalSumPool, GlobalAttnSumPool, TAGConv, APPNPConv,GlobalAvgPool, AGNNConv
from tensorflow.keras import backend
#from spektral.layers.ops import sp_matrix_to_sp_tensor
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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.982)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
from tensorflow.keras.models import Sequential
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, GlobalSumPool
from Attention import Attention
from keras_self_attention import SeqWeightedAttention as Attention
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import top_k_categorical_accuracy

#from DG_5UnifromRoIs32p_RandErasing  import DirectoryDataGenerator

#from DG_10_SliceRoIs32p_RandErasing import DirectoryDataGenerator

from DG_Birds_15SliceUnifromRoIs_GuassBlur import DirectoryDataGenerator


from AG_Net_RoiPoolingConvTF2 import RoiPoolingConv
from layer_normalization import LayerNormalization
from tensorflow.keras.regularizers import l2
#from DG_MultiScaleFiboRoIs_RandErasing import DirectoryDataGenerator


from tensorflow.keras import layers 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
#from RoiPoolingConvTF2 import RoiPoolingConv
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from SelfAttention import SelfAttention
#from SpectralNormalizationKeras import ConvSN2D
from se import squeeze_excite_block
import numpy as np
from SpatialPooling import SpatialPyramidPooling
from custom_validate_callback import CustomCallback
#from directory_data_generator_multiple_base_dirs import DirectoryDataGenerator
#from opt_dg_tf2_new import DirectoryDataGenerator

#from local_keras_applications.efficientnet import EfficientNetB4
#from local_keras_applications.efficientnet import preprocess_input as pp_input

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_input

#from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.applications.xception import preprocess_input as pp_input

import seaborn as sn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#~~~~~~~~~~~~~~~~~~~Get Paramaters From File ~~~~~~~~~~~~~~~~~~~
param_file = open("parameters.txt", "r")
params = param_file.readlines()
#Iterate through each of the parameters in the file and run them as if they were executed in this script.
#This means any script that uses the parameter.txt file share these parameters without having to change each individual script.
for line in params:
    if not line[0] == "#":
        #Remove \n from the end of each line. Necessary for string values.
        line = line.strip()
        exec(line)

#~~~~~~~~~~~~~~~~~~~Check Console Paramaters ~~~~~~~~~~~~~~~~~~~
#After the parameters.txt values have been applied, overwrite them with console-specific parameters.
#Useful for hyper-parameter searching or running experiments on different datasets without having to make a new script per experiment.
if len(sys.argv) > 2: #param 1 is file name
	total_params = len(sys.argv)
	for i in range(1, total_params, 2):
		var_name = sys.argv[i]
		new_val = sys.argv[i+1]
                #Try to make the new variable an integer/float before defaulting to string.
		try:
			exec("{} = {}".format(var_name, new_val))
		except:
			exec("{} = '{}'".format(var_name, new_val))

print("TF version:",tf.__version__)

#~~~~~~~~~~~~~~Editable parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#change the base model (e.g. InceptionV3, MobileNetV2, Xception, ResNet50)
from tensorflow.keras.applications.resnet50 import ResNet50  #as KerasModel
#all keras models use the same preprocessing function so this model name is not required to match the above model name (though it can be changed for readability).)
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_input

image_size = (224,224) #image resolution in pixels

#~~~~~~~~~~~~~~Configure Tensorflow GPU Options~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Limit the model to use only a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Stop tf2 from automatically executing tensors.
tf.compat.v1.disable_eager_execution()

#Limit the model to only use as much memory as required. By default Tensorflow will reserve 100% without needing it.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#The following should not need to be edited but can be if required. The main reason to do so would be if a dataset is labelled as train & test rather than train & val.
input_tensor = Input(shape=(image_size[0], image_size[1], 3))
train_data_dir = '{}/train/'.format(dataset_dir)
#val_data_dir = '{}/val'.format(dataset_dir)
val_data_dir = '{}/test'.format(dataset_dir)
nb_classes = len(os.listdir(train_data_dir))
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)])
verbose = True
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']

plot_output_name = "LC25k_70_30splt_MyCNN_CSGCN_15R_BSPC"


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
minSize = 2
ROIS_grid_size = 3
pool_size=4
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']
learning_rate=0.005
lr=0.005
num_rois = 15

fig_dimensions = (6, 6) #(10,7) recommended with legend
show_legend = True

def make_model(input_shape, num_classes):
    input1 = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(input1)
    x = layers.SeparableConv2D(128, 3, strides=2, padding="same")(input1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    #x = tf.keras.activations.gelu(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 768, 1024]:
        #x = layers.Activation("relu")(x)
        x = tf.keras.activations.gelu(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        #x = layers.SeparableConv2D(size, 3, padding="same")(x)

        x = layers.BatchNormalization()(x)

        #x = tf.keras.activations.gelu(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        #x = layers.SeparableConv2D(size, 3, padding="same")(x)

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
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)

    xy = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    yx = layers.AveragePooling2D(3, strides=2, padding="same")(x)
    
    x = layers.add([xy, yx])  
    x = layers.SeparableConv2D(2048, 3, padding="same")(x)
    #x = layers.SeparableConv2D(2048, 3, padding="same")(x)
        
    x = layers.BatchNormalization()(x)
    base_x = layers.Activation("relu")(x)
    #base_x = tf.keras.activations.gelu(x)

    print("final_feature.shape" , base_x.shape)
    ROIS_resolution =48
    full_img = layers.Lambda(lambda x: tf.image.resize(x,size=(ROIS_resolution, ROIS_resolution)), name='Lambda_img_1')(base_x) #Use bilinear upsampling (default tensorflow image resize) to a reasonable size

    print("full_img.shape" , full_img.shape)
    rois = layers.Input(shape=(num_rois, 4), name='RoI')
    print(rois)
    roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois)([full_img, rois])
    base_channels=2048
    feat_dim=4*4*2048
    jcvs = []

    for j in range(num_rois):
    	roi_crop = crop(1, j, j+1)(roi_pool)
    	lname = 'roi_lambda_48p'+str(j)
    	x = layers.Lambda(squeezefunc, name=lname)(roi_crop)    
    	x = layers.Reshape((feat_dim,))(x)
    	jcvs.append(x)      

    x = layers.Reshape((feat_dim,))(base_x)
    jcvs.append(x)

    jcvs = layers.Lambda(stackfunc, name='lambda_stack')(jcvs)
    print(jcvs)
    jcvs=tf.keras.layers.Dropout(0.2) (jcvs)
    print("jcvs Dropout.shape:",jcvs.shape)
    x1 = layers.TimeDistributed(layers.Reshape((pool_size,pool_size, base_channels)))(jcvs)
    print(x1)

    x1 = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time4'))(x1)
    print(x1)

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

    gc3=GCNConv(channels=1024*1, activation='relu', dropout_rate=0.30) ([p6, B_in])
    print("GNNgc3.shape:", gc3.shape)

    #gc3=layers.Add()([gc3, gc1])

    p3 = Attention(name='AttnWgt1k')(gc3)   
    #p3 = layers.GlobalAveragePooling1D()(gc3)
   
    x4 = layers.Dropout(0.2)(p3)
    # We specify activation=None so as to return logits
    bn=BatchNormalization(name='bN')(x4)
    op= layers.Dense(nb_classes, activation='softmax')(bn)

    #outputs = layers.Dense(units, activation=None)(x)
    return tf.keras.Model(inputs=[input1, rois, A_in, B_in], outputs=op)

model = make_model(input_shape=(224,224,3), num_classes=nb_classes)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()



model.optimizer.lr=learning_rate
#print("GPU-ID",str(gpu_id))
print("optimizer name", optimizer)
print("learning rate", K.eval(model.optimizer.lr))

global my_lr
my_lr = 0.00001
def epoch_decay(epoch):
    my_lr = K.eval(model.optimizer.lr)
    if (epoch %200 == 0  ) and not epoch == 0:
       my_lr = my_lr/4

    print("EPOCH: ", epoch, "NEW LR: ", my_lr)
    return my_lr

basic_schedule = LearningRateScheduler(epoch_decay)

#~~~~~~~~~~~~~~~~~Dataset Generators~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#make the generators to feed the model with train & validation data. Validation data is not augmented.
train_dg = DirectoryDataGenerator(base_directories=[train_data_dir], augmentor=True, target_sizes=image_size, preprocessors=pp_input, batch_size=batch_size, shuffle=True, verbose=verbose) #format training data
val_dg = DirectoryDataGenerator(base_directories=[val_data_dir], augmentor=False, target_sizes=image_size, preprocessors=pp_input, batch_size=batch_size, shuffle=False, verbose=verbose )#format validation data

print("Total Train images:", nb_train_samples)
print("Total validation images", nb_val_samples)

#~~~~~~~~~~~~~~~~~Model Callbacks~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Callbacks are used to add functionallity before or after each epoch.
#csv_logger = writing the training progress (epoch, accuracy, loss) to a .csv file
#checkpointer = saving the weights or full model to a file. If save_weights_only=False then the full state of the model will be stored in the file (weights + optimizer state). If it is True then only the weights are saved and the model must be recompiled to reset the optimizer learning rate. I.e when using adapative learning rate or decay, the optmizer's learning rate should be set to a different value to the initial one in the parameter.txt file.
#custom_validator = A custom callback used to evaluate the validation split every N epochs. This should be the same frequency as the checkpoint frequency for easier evaluation on the model later. This is used over the inbuilt keras method because it did not work in tensorflow 1 and for consistency this callback is still used in tensorflow 2.
#my_callbacks = a list of all the callbacks the model should be used. To add a new callback an instance of the callback must be made (like csv_logger) and the instance be added to this list.

output_model_dir = 'TrainedModels/'
metrics_dir = 'Metrics/'
training_metrics_filename = output_model_name + 'AB(Training).csv'
validation_steps=5
#Recommend using append=True to prevent any accidental overwrites of the training metric file.
csv_logger = CSVLogger(metrics_dir + training_metrics_filename, append=True)
#Recommend only saving the weights both due to the lower hard drive space taken and for custom layers weights are easier to load. False can have some conflicts if layers dont save weights correctly and make the file corrupt.
checkpointer = ModelCheckpoint(filepath = output_model_dir + output_model_name + '.{epoch:02d}.h5', verbose=1, save_weights_only=True, period=checkpoint_freq)
custom_validator = CustomCallback(val_dg, validation_freq, metrics_dir + output_model_name)

my_callbacks = [csv_logger, checkpointer, custom_validator]

model_name= output_model_name #"Food100v3_RN50_Only9RoIs_SPP35_Fourier2048_TF2_150ep"

#~~~~~~~~~~~~~~~~~~~Train & Test Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
model.fit_generator(train_dg, steps_per_epoch=nb_train_samples // batch_size,  epochs=epochs, callbacks=[checkpointer, basic_schedule, csv_logger, CustomCallback(val_dg, validation_steps, metrics_dir + model_name)]) #train and validate the model

#save the final model (both the weights and optimizer state).
model.save(output_model_dir + output_model_name + ".h5")

#~~~~~~~~~~~~~~~~~Release GPU Resources~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#There are some situations where tensorflow fails to release resources automatically (especially in tensorflow 2). The following 2 lines resolve this.
#del model
#backend.clear_session()

#model.fit_generator(train_dg, steps_per_epoch=nb_train_samples // batch_size,  epochs=epochs, callbacks=[checkpointer, csv_logger, CustomCallback(val_dg, validation_steps, model_name)]) #train and validate the model
#model.save(output_model_dir + model_name + ".h5") #save the final model

#~~~~~~~~~~~~~~~~~Release GPU Resources~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

loss, acc1 = model.evaluate_generator(val_dg)
print("top 1 acc:", acc1)

top_N_acc = 3
def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=top_N_acc)
metric = top_k_accuracy


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[metric])

#output
##evaluate all images in the data generator
predictions = model.predict_generator(val_dg)

loss, acc3 = model.evaluate_generator(val_dg)
print(f"top {top_N_acc} accuracy:", acc3)

y_true = []
y_proba = []
y_true_max = []
y_pred_max = []

for index in range(predictions.shape[0]//batch_size):
    _, labels = val_dg.__getitem__(index)
    for i in range(batch_size):
        pred = predictions[(index*batch_size)+i]
        y_true_max.append(np.argmax(labels[i]))
        y_pred_max.append(np.argmax(pred))
        y_true.append(labels[i])
        y_proba.append(pred)


##output
y_pred_max = np.array(y_pred_max)
y_true_max = np.array(y_true_max)


from sklearn.metrics import classification_report
print(classification_report(y_true_max , y_pred_max))


cm_ = confusion_matrix(y_true_max, y_pred_max)
cm_ = cm_ / cm_.astype(float).sum(axis=1)

df_cm = pd.DataFrame(cm_, index = [str(i) for i in range(nb_classes)],
                  columns = [str(i) for i in range(nb_classes)])

plt.figure(figsize = fig_dimensions)
g = sn.heatmap(df_cm, cmap='Blues', annot=True, fmt=".4f",  annot_kws={'size': 10}, cbar=show_legend)

plt.savefig("{}.png".format(plot_output_name), bbox_inches='tight', pad_inches=0, dpi=300)

print(cm_)

from sklearn.metrics import RocCurveDisplay, roc_curve
y_pred_max = np.array(y_pred_max)
y_true_max = np.array(y_true_max)

print("\ny_pred_max\n", y_pred_max)

print("\ny_true_max\n", y_true_max )


print("Done!")


