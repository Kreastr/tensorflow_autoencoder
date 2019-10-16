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

def getLayer(x, numIn, numOut, weights_list, actfun=tf.nn.sigmoid):

    h = tf.Variable(tf.random_normal([numIn, numOut]))
    b = tf.Variable(tf.random_normal([numOut]))
    weights_list.append(h)
    layer = actfun(tf.add(tf.matmul(x, h), b))
    layer.my_h = h
    layer.my_b = b
    layer.my_parent = x
    return layer

def getScaler(x_target, x):
    return tf.div(
       0.00001 + x,
       tf.reduce_max(0.00001 +x_target)
     )

def getMinMax(x, inp_size, window=20):
    mdl = TfModel()
    tmp = tf.reduce_mean(tf.reshape(x, [-1, window, int(inp_size / window)]), 1)
    mdl.input_avgs = tf.tile(tmp, [1, window])
    mdl.input_deltas = tf.concat([x - mdl.input_avgs, tmp], 1)

    mdl.min_deltas = tf.reduce_min(mdl.input_deltas, 0)
    mdl.max_deltas = tf.reduce_max(mdl.input_deltas, 0)
    return mdl

def getScalerNoHinge(x_target, x):
    return tf.div(
       x,
       tf.reduce_max(0.00001 +x_target)
     )

def getModel(X, Y, noiseLevel, minsc, maxsc, inp_size, num_classes=3, windowsize=20, nh=[200, 50], lr=0.01, vae_batch=256):
    mdl = TfModel()
    mdl.classification_space_size = inp_size
    mdl.learning_rate = lr
    # Network Parameters
    mdl.num_hidden = nh # 1st layer num features
    mdl.num_hidden_dec = mdl.num_hidden[-2::-1] # 1st layer num features
    mdl.num_hidden_dec.append(mdl.classification_space_size  + int(mdl.classification_space_size/windowsize))

    num_input = mdl.classification_space_size # MNIST data input (img shape: 28*28)

    def getNet(x, inp_size, hidden_sizes, weights_list,af=tf.nn.sigmoid):
        # Encoder Hidden layer with sigmoid activation #1

        input = x
        input_size = inp_size
        for size in hidden_sizes:
            input = getLayer(input, input_size, size, weights_list, actfun=af)
            input_size = size

        return input

    # Construct model
    noisedX = X+noiseLevel*tf.random_normal(tf.shape(X), 0, 1, dtype=tf.float32)

    tmp = tf.reduce_mean(tf.reshape(noisedX, [-1, windowsize, int(mdl.classification_space_size / windowsize)]), 1)
    mdl.input_avg = tmp
    mdl.input_avgs = tf.tile(tmp, [1, windowsize])
    mdl.input_deltas = tf.concat([noisedX - mdl.input_avgs,tmp], 1)
    target = tf.divide((mdl.input_deltas - minsc),
                       0.0000001 + maxsc - minsc)
    mdl.target = target
    #target = tf.divide((mdl.input_avg-tf.reduce_min(mdl.input_avg, 0)),0.00000001+(tf.reduce_max(mdl.input_avg, 0)-tf.reduce_min(mdl.input_avg, 0)))
    listenc = []
    mdl.encoder_op_mean = getNet(target, mdl.classification_space_size + int(mdl.classification_space_size/windowsize), mdl.num_hidden, listenc, af=tf.nn.sigmoid)
    #mdl.encoder_op_var = ( getNet(X-0.5, mdl.classification_space_size, mdl.num_hidden, af=tf.nn.sigmoid))
    #samples = tf.random_normal([vae_batch, mdl.num_hidden[-1]], 0, 0.3, dtype=tf.float32)

    #target_2 = tf.divide((mdl.encoder_op_mean- tf.reduce_min(mdl.encoder_op_mean, 0)),
                       #0.00000001 + (tf.reduce_max(mdl.encoder_op_mean, 0) - tf.reduce_min(mdl.encoder_op_mean, 0)))
    mdl.enc_smpl = mdl.encoder_op_mean#target_2-0.5#mdl.input_avg#tf.concat([mdl.encoder_op_mean, mdl.input_avg], 1)#mdl.encoder_op_mean #+ mdl.encoder_op_var*samples
    listdec = []
    mdl.decoder_op = (getNet(mdl.encoder_op_mean-0.5, mdl.num_hidden[-1], mdl.num_hidden_dec, listdec, af=tf.nn.sigmoid))
    #mdl.num_hidden[-1]++int(mdl.classification_space_size / 20)
    listclass = []
    mdl.classifier_op = (getNet(mdl.enc_smpl, mdl.num_hidden[-1] , [num_classes], listclass, af=tf.nn.sigmoid))

    mdl.ohY = tf.one_hot(Y, num_classes)
    #tf.reduce_mean(tf.pow(mdl.classifier_op - mdl.ohY, 2))#
    # Prediction
    y_pred = mdl.decoder_op
    # Targets (Labels) are the input data.



    mdl.delta_avg = tf.reduce_mean(tf.abs(mdl.input_deltas))
    y_true = mdl.input_deltas

    # Define loss and optimizer, minimize the squared error

    #mdl.compression_error_avg = tf.reduce_mean(tf.reshape(y_true - y_pred, [-1, 20, int(mdl.classification_space_size/20)]), 1)
    #z_mean = mdl.encoder_op_mean
    #z_stddev = mdl.encoder_op_var
    mdl.latent_loss = tf.reduce_mean(-tf.log(tf.reduce_max(mdl.encoder_op_mean,0)-tf.reduce_min(mdl.encoder_op_mean,0)+0.0000001))

    reg_lossesencdec = 0
    reg_lossesencclass = 0
    for w in listenc:
        reg_lossesencdec += tf.nn.l2_loss(w)
        reg_lossesencclass += tf.nn.l2_loss(w)
    for w in listdec:
        reg_lossesencdec += tf.nn.l2_loss(w)
    for w in listclass:
        reg_lossesencclass += tf.nn.l2_loss(w)

    reg_constant = 0.000001  # Choose an appropriate one.
    mdl.reglossencdec = tf.constant(0.0)#reg_constant*tf.reduce_mean(reg_lossesencdec)
    mdl.reglossencclass= tf.constant(0.0)#reg_constant * tf.reduce_mean(reg_lossesencclass)

    mdl.class_loss = mdl.latent_loss+tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=mdl.classifier_op, labels=mdl.ohY))+ mdl.reglossencclass
    mdl.class_opt = tf.train.RMSPropOptimizer(0.0001 * mdl.learning_rate, momentum=0.99, decay=0.9).minimize(
        mdl.class_loss)

    mdl.loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))+mdl.latent_loss+ mdl.reglossencdec
    mdl.optimizer = tf.train.RMSPropOptimizer(mdl.learning_rate, momentum=0.9, decay=0.99).minimize(mdl.loss)
    mdl.auc = tf.metrics.auc(mdl.ohY, mdl.classifier_op)
    mdl.acc = tf.metrics.accuracy(Y, tf.argmax(mdl.classifier_op, 1))
    return mdl


