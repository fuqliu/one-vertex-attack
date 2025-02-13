from models.layers import *
from os.path import join as pjoin
from attack.model_base import *

import tensorflow as tf 
import numpy as np


def build_noise_model(inputs, n_his, Ks, Kt, blocks, keep_prob, theta, batch_size, n, noise_para):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    :param theta: reglarization parameter
    '''
    #added by fuqiang liu
    xx = inputs[:, 0:n_his, :, :]
    x = xx+0.00000000001
    # define universal adversary
    universal_noise = tf.compat.v1.get_variable(name='unoise', shape=[1,n_his, n, 1], dtype=tf.float32)
    tf.add_to_collection(name='unoise', value=universal_noise)
    # add noise
    x = x + tf.tile(universal_noise,[tf.shape(x)[0],1,1,1])
    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        rate = 1 - keep_prob
        x = st_conv_block(x, Ks, Kt, channels, i, rate, act_func='GLU')
        Ko -= 2 * (Ks - 1)

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    # targeted label
    target = -3*tf.sign(inputs[:, n_his:n_his + 1, :, :])-0.999999999999999
    # define loss
    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='target_loss',
                         value=tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='noise_loss',
                         value=2*tf.nn.l2_loss(universal_noise))
    train_loss = tf.nn.l2_loss(y - target) + theta*noise_l2_loss(universal_noise,noise_para,n_his*n)
#    train_loss = tf.nn.l2_loss(y - target) 
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred
