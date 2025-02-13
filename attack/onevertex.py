'''
one vertex 
'''
from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from models.layers import *
from os.path import join as pjoin
from attack.variable_restore import *
from attack.model_base import *

import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
import time
import numpy as np


def build_generator_model(inputs, n, n_his, Ks, Kt, blocks, keep_prob, batch_size, generator_channel, noise_para, theta):
#   function used to build the noise generator
#   :para inputs: input data
#   :para n: number of route
#   :para n_his: number of historical record
#   :para Ks: size of spatial conv kernel
#   :para Kt: size of temporal kernel
#   :para keep_prod: placeholder
#   :para batch_size:batch_size
#   :para generator_channel: number of generator channels

    x = inputs[:, 0:n_his, :, :]
    
    # define generator
    generator_weights_1 = tf.compat.v1.get_variable(name='generator_weights_1', shape=[n_his, generator_channel], dtype=tf.float32)
    tf.add_to_collection(name='generator_weights_1', value=generator_weights_1)
    generator_weights_2 = tf.compat.v1.get_variable(name='generator_weights_2', shape=[generator_channel, generator_channel], dtype=tf.float32)
    tf.add_to_collection(name='generator_weights_2', value=generator_weights_2)
    generator_weights_3 = tf.compat.v1.get_variable(name='generator_weights_3', shape=[generator_channel, n_his], dtype=tf.float32)
    tf.add_to_collection(name='generator_weights_3', value=generator_weights_3)

    # generate noise
    noise = tf.tensordot(x,generator_weights_1,[[1],[0]])
    noise = tf.tensordot(noise,generator_weights_2,[[3],[0]])
    noise = tf.tensordot(noise,generator_weights_3,[[3],[0]])
    noise = tf.transpose(noise,perm=[0,3,1,2])
    tf.add_to_collection(name='noise', value=noise)

    # add noise
    x = x + noise
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
                         value=2*tf.nn.l2_loss(noise))
    
    train_loss = tf.nn.l2_loss(y - target) + theta*noise_l2_loss_2(noise, noise_para)
    
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred

def onevertex_generator_train(inputs, blocks, args, load_path, sum_path='./output/onevertex'):
# build up a new graph, and create a session as: sess
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    theta = args.theta
    noise_para = args.noise_para
    generator_channel = args.generator_channel

    # Placeholder for model training
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    # get the check point of the pretrained model
    model_path = tf.train.get_checkpoint_state(load_path)
    pretrained_model = model_path.model_checkpoint_path

    # Define model loss                          
    train_loss, pred = build_generator_model(x, n, n_his, Ks, Kt, blocks, keep_prob, batch_size, generator_channel, noise_para, theta)
    
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)
    target_loss = tf.add_n(tf.get_collection('target_loss'))
    tf.summary.scalar('target_loss', target_loss)
    noise_loss = tf.add_n(tf.get_collection('noise_loss'))
    tf.summary.scalar('noise_loss', noise_loss)
    
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1

    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)

    # freeze variables
    trainable_vars = tf.trainable_variables()
    universal_noise_var_list = [t for t in trainable_vars if t.name.startswith(u'generator_weights')]
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(loss = train_loss, var_list = universal_noise_var_list)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss, var_list = universal_noise_var_list)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()
    

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        # initialize all variables
        variables = tf.global_variables()
        sess.run(tf.global_variables_initializer())

        # get the trained variables
        var_keep_dic = get_variable_in_checkpoint_file(pretrained_model)
        variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, pretrained_model)
        print("=========loaded===========.")

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss, target_loss, noise_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: train_loss-{loss_value[0]:.3f}, copy_loss-{loss_value[1]:.3f}, target_loss-{loss_value[2]:.3f}, noise_loss-{loss_value[3]:.3f}')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'onevertex', './output/onevertex/models/')
        writer.close()
    print('Training model finished!')

def onevertex_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()

    x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

    adversary = np.array(get_noise_onevertex(x_test, n_his))

    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test[:,0:n_his,:,:] = x_test[:,0:n_his,:,:] + adversary[0,:,:,:,:]
        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)

        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
    print(f'the energy of adversary is {np.sum(adversary**2)/np.size(adversary)}')

def get_noise_onevertex(x_test, n_his, load_path = './output/onevertex/models/'):

    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        
        noise = test_graph.get_collection('noise')
        adversary = test_sess.run(noise,feed_dict={'data_input:0': x_test[:,0:n_his+1,:,:], 'keep_prob:0': 1.0})
    return adversary
