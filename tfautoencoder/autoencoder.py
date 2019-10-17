#
#    Copyright 2018 Aleksei Romanenko
#
#    This file is part of Tensorflow Autoencoder.
#
#    Tensorflow Autoencoder is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Tensorflow Autoencoder is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Tensorflow Autoencoder.  If not, see <https://www.gnu.org/licenses/>.

#from transliterate import translit, get_available_language_codes

import tensorflow as tf

class TfModel: pass
from multiprocessing import Process, Queue

def getLayer(x, numIn, numOut, weights_list, actfun=tf.nn.sigmoid, name=None):

    h = tf.Variable(tf.random_normal([numIn, numOut]))
    b = tf.Variable(tf.random_normal([numOut]))
    weights_list.append(h)
    layer = actfun(tf.add(tf.matmul(x, h), b), name=name)
    layer.my_h = h
    layer.my_b = b
    layer.my_parent = x
    return layer

def getScaler( x):
    return tf.div(
       0.00001 + x,
       tf.reduce_max(0.00001 + x)
     )

def getMinMax(x, inp_size, window=20):
    mdl = TfModel()
    tmp = tf.reduce_mean(tf.reshape(x, [-1, window, int(inp_size / window)]), 1)
    mdl.input_avgs = tf.tile(tmp, [1, window])
    mdl.input_deltas = tf.concat([x - mdl.input_avgs, tmp], 1)

    mdl.min_deltas = tf.reduce_min(mdl.input_deltas, 0)
    mdl.max_deltas = tf.reduce_max(mdl.input_deltas, 0)
    return mdl

def getScalerNoHinge(x):
    return tf.div(
       x,
       tf.reduce_max(0.00001 + x)
     )

def getModel(X, inp_size, num_classes=3, nh=[200, 50], lr=0.01, vae_batch=256):
    mdl = TfModel()
    mdl.classification_space_size = inp_size
    mdl.learning_rate = lr
    # Network Parameters
    mdl.num_hidden = nh # 1st layer num features
    mdl.num_hidden_dec = mdl.num_hidden[-2::-1] # 1st layer num features
    mdl.num_hidden_dec.append(mdl.classification_space_size)

    num_input = mdl.classification_space_size # MNIST data input (img shape: 28*28)

    def getNet(x, inp_size, hidden_sizes, weights_list,af=tf.nn.sigmoid,name_prefix=""):
        # Encoder Hidden layer with sigmoid activation #1

        input = x
        input_size = inp_size
        for i, size in enumerate(hidden_sizes):
            input = getLayer(input, input_size, size, weights_list, actfun=af, name=f"{name_prefix}layer_output_{i}")
            input_size = size

        return input

    # Construct model
    target = X
    mdl.target = target
    
    listenc = []
    mdl.encoder_op = getNet(target, mdl.classification_space_size, mdl.num_hidden, listenc, af=tf.nn.sigmoid, name_prefix="encoder_")
    
    mdl.enc_smpl = mdl.encoder_op
    
    listdec = []
    mdl.decoder_op = (getNet(mdl.encoder_op-0.5, mdl.num_hidden[-1], mdl.num_hidden_dec, listdec, af=tf.nn.sigmoid, name_prefix="decoder_"))
    
    inputs = mdl.target
    restored = mdl.decoder_op
    
    # Define loss and optimizer, minimize the squared error

    mdl.latent_loss = tf.reduce_mean(-tf.log(tf.reduce_max(mdl.encoder_op,0)-tf.reduce_min(mdl.encoder_op,0)+0.0000001))

    reg_lossesencdec = 0
    for w in listenc:
        reg_lossesencdec += tf.nn.l2_loss(w)
        
    for w in listdec:
        reg_lossesencdec += tf.nn.l2_loss(w)

    reg_constant = 0.000001  # Choose an appropriate one.
    mdl.reglossencdec = tf.constant(0.0)#reg_constant*tf.reduce_mean(reg_lossesencdec)

    mdl.loss = tf.reduce_mean(tf.pow(inputs - restored, 2))+mdl.latent_loss+ mdl.reglossencdec
    mdl.optimizer = tf.train.RMSPropOptimizer(mdl.learning_rate, momentum=0.9, decay=0.99).minimize(mdl.loss)
    return mdl


