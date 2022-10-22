#!/usr/bin/env python
# coding: utf-8



#from matplotlib import pyplot as plt
from cmath import nan
import random as rnd
import numpy as np
import collections
import os
import math

from utils_quantize import *
#from kmeans import *
from models.cifar10_models import build_model


# tf and keras
import tensorflow as tf
#import pyclustering
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
#from sklearn.cluster import KMeans
#from pyclustering.cluster.kmeans import kmeans
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.integrate import quad
from csvec import CSVec
import torch
import math
#------------------------------
# DNN settings
#learning_rate = 0.0001
learning_rate = 0.01
epochs = 3



rnd.seed(5)
np.random.seed(5)
tf.random.set_seed(5)       # seed

#torch.cuda.empty_cache()
batch = 32                  # VGG 16    other 32
iterations = 10
number_of_users = 2

#sparsification_percentage = 60


# sparse_gradient[0].shape

layers_to_be_compressed=np.array([6,12,18,24,30,36,42])

#compression_type="uniform scalar"
#compression_type="uniform scalar with memory"
#compression_type="k-means"
#compression_type="k-means with memory"
#compression_type="sketch"
compression_type="weibull"

#compression_type="no compression"

#compression_type="no compression with float16 conversion"

#------------------------------
# channel settings





def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation


# In[10]:


def top_k_sparsificate_model_weights_tf(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
    print("total number of parameters:",len(tmp_list))
    #TODO
    # same as weight.reshape.size[0] ? better make it more general
    # write as in 183
    k_th_element = tmp_list[int(fraction*len(tmp_list))-1] # 552874 is the number of parameters of the CNNs!       23608202:Res50   0.0004682019352912903
    new_weights = []
    #new_weight = []
    for el in weights:
        '''
        original_shape = el.shape
        reshaped_el = el.reshape((-1))
        for i in range(len(reshaped_el)):
            if abs(reshaped_el[i]) < k_th_element:
                reshaped_el[i] = 0.0
        new_weights.append(reshaped_el.reshape(original_shape))
        '''
        mask = tf.math.greater_equal(tf.math.abs(el), k_th_element)
        new_w = tf.multiply(el, tf.cast(mask, weights[0]))
        new_weights.append(new_w.numpy())
    '''    
    # 60% test
    num=0 
    for el in new_weights:
        num = num + np.count_nonzero(el)
    print("percentage:", num/len(tmp_list))
    '''
    return new_weights

def pdf_doubleweibull(x, a, m, scale=1):
  return stats.dweibull.pdf(x,a,m,scale)

def update_centers_magnitude_distance_weibull(data, R, iterations_kmeans):
    M = QUANTIZATION_M
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.dweibull.fit(data_normalized)
    print(a,m,b)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)

def pdf_gennorm(x, a, m, b):
  return stats.gennorm.pdf(x,a,m,b)

def update_centers_magnitude_distance(data, R, iterations_kmeans):
    #TODO: allow change of m
    M = QUANTIZATION_M
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.gennorm.fit(data_normalized)
    print(a,m,b)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
            #if (np.abs(integ_nom_update)<0.0000000001) or (np.abs(integ_denom_update)<0.0000000001):
            #    centers_update[j + 1] = 0
            #else:
            #    centers_update[j + 1] = np.divide(integ_nom_update, integ_denom_update)  # integ_denom_update+eplison
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)

def fp8_152_bin_edges(exponent_bias=15):
    bin_centers = np.zeros(247,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(247, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(31):
            for f in range(4):
                if e != 0:
                    exponent = e - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f, 'b').zfill(2)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f,'b').zfill(2)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict[0.0] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    #print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict


def print_model_size(mdl):
    #torch.save(mdl.state_dict(), "tmp.pt")
    mdl.save_weights('./checkpoints/tmp')
    size = os.path.getsize('./checkpoints/tmp.data-00000-of-00001')
    print("%.2f MB" %(size /1e6))
    os.remove('./checkpoints/tmp.data-00000-of-00001')



# In[11]: About Data
classes = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck",
}

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
num_classes = len(classes)

# normalize to one
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# categorical loss enropy
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)




# In[5]: About Model
initializer = tf.keras.initializers.HeUniform(seed=5)
model = tf.keras.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same', input_shape=(32, 32, 3)),
  layers.BatchNormalization(),
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.3),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5)),
  layers.BatchNormalization(),
  layers.Dropout(0.5),
  layers.Dense(10, activation='softmax'),
])


#model = build_model['ResNet18'](X_train)
#model._name = 'resnet18'

if model._name == 'VGG16':
    opt = Adam(learning_rate=0.00005)
    number_threshold = 500000
elif model._name == 'resnet18':
    opt = Adam()
    number_threshold = 100000
else:
    # DNN
    opt = Adam(learning_rate=0.0001)
    number_threshold = 1000

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()




num_comp = 0
large_layers = []
for i in range(len(model.get_weights())):
    if model.get_weights()[i].size > number_threshold:
        large_layers.append(i)
        num_comp = num_comp + model.get_weights()[i].size
layers_to_be_compressed = np.asarray(large_layers)
print("layers to be compressed:", layers_to_be_compressed)
print("Compressing number:", num_comp)    
# layers_to_be_compressed=np.array([6,12,18,24,30,36,42])   DNN
# layers to be compressed: [ 72  78  96 114 132 144 150 156 158 168 174 180 186 192 198 204 210 216 222 228 234 240 246 252 258 264 270 272 282 288 294 300 306 312]
# In[]:

# FL setting 
size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users,size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users,size_of_user_ds,10))
for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    

import argparse

parser = argparse.ArgumentParser(description='Federared Learning Parameter')
parser.add_argument('--R', type=int, default=1,
                    help='How many bit rate')
parser.add_argument('--M', type=int, default=0,
                    help='M for K-means Quantizer')
parser.add_argument('--P', type=int, default=60,
                    help='Sparsification percentage')

args = parser.parse_args()




# Parameter
QUANTIZATION_M = args.M
BIT_RATE = args.R
sparsification_percentage = args.P

d=1
# # indexed by dimenions of the input, so far we go through each layer separately
rate = np.array([d])
rate[0] = BIT_RATE


# # this is an array too, 
# # indexed as [rate][memory]
# c_scale = np.ones([10,10])

# -----------------------------------------------------------------------------
iter = 0



#constructing the memory array
w_before_train = model.get_weights()
memory = [np.subtract(w_before_train[i], w_before_train[i]) for i in range(len(w_before_train))]
memory_array = np.tile(memory, (iterations+1,number_of_users,1))

#Beta = [0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0, 0]
#Beta = [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0]


from datetime import datetime
timeObj = datetime.now().time()
timeStr = timeObj.strftime("%H_%M_%S_%f")
out_file = "accuracy-"+compression_type+"-R"+str(BIT_RATE)+"-M" +str(QUANTIZATION_M)+"-fixed-seed9"


print(compression_type)
print("M =", QUANTIZATION_M)
print("Rate =", rate[0])
gradient_list = []

with open(out_file + timeStr + '.txt', "w") as outfile:
    for _ in range(iterations):
      print('')
      print("************ Iteration " + str(_) + " ************")
      _, accuracy = model.evaluate(X_test, Y_test)
      print('Test accuracy BEFORE training is', accuracy)
      #model.save("multi-user model-before training-TCQ_iter"+str(iter)+".h5")
      iter = iter + 1
      wc = model.get_weights()
      sum_terms = []

      #beta = Beta[iter-1]
      beta=0.3
      #print("beta: ", beta)

      for i in range(number_of_users):
        X_train_u = train_data_X[i]
        Y_train_u = train_data_Y[i]
        np.random.seed(5)
        shuffler = np.random.permutation(len(X_train_u))
        X_train_u = X_train_u[shuffler]
        Y_train_u = Y_train_u[shuffler]
        X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)

        print('user->',i)
        print("beta: ", beta)
        #print(len(X_train_u))
        history = model.fit(x=X_train_u,y=Y_train_u,
                              epochs = epochs,
                              batch_size = batch,
                              validation_data=(X_validation_u, Y_validation_u),
                              shuffle=False
                              )
        #model.save("multi-user model-after training-TCQ_iter" + str(iter)+"_user"+str(i) + ".h5")

        # check model size
        #print_model_size(model)
        # compare model

        _, accuracy = model.evaluate(X_test, Y_test)


        # TODO
        # i'd use a more meaningful name ^_^
        # Communication PS->clients
        wu = model.get_weights()


        nu = len(Y_train_u)+len(Y_validation_u)
        frac = nu/len(Y_train)

        # approx gradient with model difference
        gradient = [np.subtract(wu[i], wc[i]) for i in range(len(wu))]
        #print('sparse level:', sparsification_percentage/100)
        #sparse_gradient = top_k_sparsificate_model_weights_tf(gradient, sparsification_percentage/100)
        print('sparse level:', sparsification_percentage/(BIT_RATE*100))
        sparse_gradient = top_k_sparsificate_model_weights_tf(gradient, sparsification_percentage/(BIT_RATE*100))
        #sparse_gradient = gradient

        #for j in range(len(sparse_gradient)):
        layer_index = 1
        for j in layers_to_be_compressed:
          #np.savetxt(outfile, [np.reshape(gradient[j],(np.size(gradient[j],))), ], fmt='%10.3e', delimiter=',')
          # the size of this is 44
          # I would skip all the layers that have a small size.
          # only compress the ones in layers_to_be_compressed
          gradient_shape = np.shape(sparse_gradient[j])
          gradient_size = np.size(sparse_gradient[j])
          gradient_reshape = np.reshape(sparse_gradient[j],(gradient_size,))
          non_zero_indices = tf.where( gradient_reshape != 0).numpy()

          #reshaping the memory
          memory_shape = np.shape(memory_array[iter-1,i,j])
          memory_size = np.size(memory_array[iter-1,i,j])
          memory_reshape = np.reshape(memory_array[iter-1,i,j],(memory_size,))


          print("Layer",j,": entries to compress:",non_zero_indices.size, "total # entries:", gradient_size )


          if compression_type == "sketch":
                sparse_g_tensor = torch.tensor(sparse_gradient[j]).to(device='cuda')
                sparse_g_tensor_flatten = sparse_g_tensor.view(-1)
                # perform sketch to this seq
                # should we perform sketch on weight tensors or on non-zero indices?
                num_rows = 5
                num_cols = math.floor(gradient_size/(1*num_rows))        # Change to change the rate?   10
                sketch = CSVec(d=gradient_size, c=num_cols, r=num_rows)   # device="cpu", numBlocks=1
                sketch.accumulateVec(sparse_g_tensor_flatten)
                seq_enc = sketch.table

                # unsketch
                #num_nonzero = gradient_size
                num_nonzero = len(non_zero_indices)
                sketch = CSVec(d=gradient_size, c=num_cols, r=num_rows)
                sketch.accumulateTable(seq_enc)
                seq_dec = sketch.unSketch(k=num_nonzero)  
                # compare  seq_dec with sparse_g_tensor_flatten
                # assert match_shape

                sparse_gradient[j] = torch.reshape(seq_dec, gradient_shape).cpu().numpy()
                continue


          #SR2SS
          # i would say >1000, no need to worry about the small dimensions her
          if (non_zero_indices.size > 1):
              seq = gradient_reshape[np.transpose(non_zero_indices)[0]]
              mem_seq = memory_reshape[np.transpose(non_zero_indices)[0]]


              if  compression_type=="uniform scalar":

                  seq_enc, uni_max, uni_min= compress_uni_scalar(seq, rate)


                  seq_dec = decompress_uni_scalar(seq_enc, rate, uni_max, uni_min)

              elif  compression_type=="gaussian scalar":
                 seq_enc, mu, s = gaussian_compress(seq, rate[0])
                 seq_dec = decompress_gaussian(seq_enc, mu, s)

              elif compression_type=="k-means":
                      thresholds, quantization_centers = update_centers_magnitude_distance(data=seq, R=rate[0],  iterations_kmeans=100)
                      thresholds_sorted = np.sort(thresholds)
                      labels = np.digitize(seq,thresholds_sorted)
                      index_labels_false = np.where(labels == 2**rate[0])
                      labels[index_labels_false] = 2**rate[0]-1
                      seq_dec = quantization_centers[labels]

              elif compression_type=="weibull":
                      thresholds, quantization_centers = update_centers_magnitude_distance_weibull(data=seq, R=rate[0],  iterations_kmeans=100)
                      thresholds_sorted = np.sort(thresholds)
                      labels = np.digitize(seq,thresholds_sorted)
                      index_labels_false = np.where(labels == 2**rate[0])
                      labels[index_labels_false] = 2**rate[0]-1
                      seq_dec = quantization_centers[labels]

              elif compression_type == "k-means with memory":
                  #beta = 0.5
                  seq_to_be_compressed = seq+beta*mem_seq
                  thresholds, quantization_centers = update_centers_magnitude_distance(data=seq_to_be_compressed, R=rate[0],  iterations_kmeans=100)
                  thresholds_sorted = np.sort(thresholds)
                  labels = np.digitize(seq, thresholds_sorted)
                  index_labels_false = np.where(labels == 2 ** rate[0])
                  labels[index_labels_false] = 2 ** rate[0] - 1
                  seq_dec = quantization_centers[labels]
                  seq_error = beta*mem_seq+seq_to_be_compressed-seq_dec
                  np.put(memory_reshape, np.transpose(non_zero_indices)[0], seq_error)

                  memory_array[iter,i,j] = memory_reshape.reshape(memory_shape)
                  # SR2SS
                  # need to stare overything: see how it changes over layer and over time

              elif compression_type == "optimal compression":
                  seq_enc , mu, s = optimal_compress(seq,rate)
                  seq_dec = decompress_gaussian(seq_enc, mu, s)


              elif compression_type == "no compression":
                  seq_dec = seq

              elif compression_type == "no compression with float16 conversion":
                  seq_dec = seq.astype(np.float16)

              elif compression_type == "no compression with float8 conversion":
                  fp8_bin_centers, fp8_bin_edges, fp8_dict = fp8_152_bin_edges()
                  indices = np.digitize(seq, fp8_bin_edges)
                  seq_dec = fp8_bin_centers[indices]


              # compress_decompress(type='TCQ')

              #plot the histogram of data



              #saving the histogram after compression
              #dec_max = np.amax(seq_dec)
              #dec_min = np.amin(seq_dec)
              #step_size = (dec_max - dec_min) / 100
              #bins_array_dec = np.arange(dec_min, dec_max, step_size)
              #hist_after, bin_edges_after = np.histogram(seq_dec, bins=bins_array_dec)


              #saving histogram after compression
              #if ((j== 6) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[1],header='#layer6-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after)-1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e', delimiter=',')
              #if ((j== 24) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[2],header='#layer24-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after)-1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e',delimiter =',')
              #if ((j == 42) & (i == 0) & (iter == 10)):
              #    np.savetxt(outfile, [3], header='#layer42-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after) - 1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index], hist_after[bin_index]],], fmt='%10.3e',delimiter=',')



              #unique_labels, unique_indices, counts = np.unique(seq_dec,return_index=True,return_counts=True)
              #if ((j== 12) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[1],header='#layer12-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e',delimiter=',')
              #if ((j== 24) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[2],header='#layer24-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e', delimiter =',')
              #if ((j == 42) & (i == 0) & (iter == 10)):
              #    np.savetxt(outfile, [3], header='#layer42-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],], fmt='%10.3e',delimiter=',')
              #np.savetxt(outfile, [bin_edges_after])
              #np.savetxt(outfile, [hist_after])
              #fig = plt.figure()
              #ax = fig.add_subplot(1, 1, 1)
              #ax.hist(seq_dec, bins=bins_array)
              #plt.xlabel('bins')
              #plt.ylabel('histogram of quantized data')
              #fig.savefig('hist-after compression-'+'Iter'+str(iter)+'-Layer'+ str(j)+'.png')


              np.put(gradient_reshape, np.transpose(non_zero_indices)[0], seq_dec)

              sparse_gradient[j] = gradient_reshape.reshape(gradient_shape)
              layer_index = layer_index+1

        #user_gradient = [np.add(wc[i], sparse_gradient[i]) for i in range(len(sparse_gradient))]
        #gradient_list.append(user_gradient)
        
        # this is the PS part
        # Communication clients to PS
        sum_terms.append(np.multiply(frac,sparse_gradient))
        model.set_weights(wc)

      
      #user0_grad = gradient_list[0]
      #user1_grad = gradient_list[1]
      #user0_grad_half = [np.multiply(user0_grad[i], 0.5) for i in range(len(user0_grad))]
      #user1_grad_half = [np.multiply(user1_grad[i], 0.5) for i in range(len(user1_grad))]
      #avg = [np.add(user0_grad_half[i], user1_grad_half[i]) for i in range(len(user0_grad_half))]

      update = sum_terms[0]
      for i in range(1, len(sum_terms)): # could do better...
        tmp = sum_terms[i]
        update = [np.add(tmp[j], update[j]) for j in range(len(update))]
      new_weights = [np.add(wc[i], update[i]) for i in range(len(wc))]
      model.set_weights(new_weights)

      # check model size
      #print_model_size(model)
      # compare model

      # check test accuracy
      results = model.evaluate(X_test, Y_test)
      # check the performance at the PS, monitor the noise
      print('Test accuracy AFTER PS aggregation',results[1])
      #np.savetxt(outfile, [[int(iter),results[1]],],fmt='%10.3e',delimiter =',')
      np.savetxt(outfile, [results[1]],fmt='%10.3e')






# In[ ]:




