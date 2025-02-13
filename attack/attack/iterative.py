from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import copy

def clip(inputs, new_inputs, clip_e):
    temp = np.zeros((np.size(inputs,0), np.size(inputs,1), np.size(inputs,2), np.size(inputs,3), 2))
    temp[:,:,:,:,0] = inputs + clip_e
    temp[:,:,:,:,1] = np.max([new_inputs,inputs-clip_e],0)
    clip_x = np.min(temp,4)
    return clip_x

def ite_generate_adversary(load_path, x_test, n_his, n_pred, BIM_para, BIM_config, clip_epoch, clip_e):
    '''
    function to generate adversary based on the Basic Iterative Method
    :param load_path: path of the graph and model
    :param x_test: test data
    :param n_his: length of historical data
    :param n_pred: time steps of prediction
    :param BIM_para: l-infinite of the adversary
    :param BIM_config: types of method ["none","back","target","target2"]
    :param clip_para: parameters of Clip
    '''
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as sess:

        n_frame = n_his + n_pred

        saver.restore(sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')
        
        gradient = test_graph.get_collection('gradient')
                
        clip_x = np.copy(x_test)
        
        for epoch in range(clip_epoch):
            dif = np.zeros((np.size(x_test, 0), n_his, np.size(x_test,2), np.size(x_test,3)))
            
            for i in range(np.size(x_test, 0)-n_frame):

                if BIM_config == 'none':
                    inputs = copy.deepcopy(clip_x[i:i+1, 0:n_his+1, :, :])
                elif BIM_config == 'back':
                    inputs = copy.deepcopy(clip_x[i:i+1, 0:n_his+1, :, :])
                    inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
                elif BIM_config == 'target2':
                    inputs = copy.deepcopy(clip_x[i:i+1, 0:n_his+1, :, :])
                    inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
                    inputs[np.where(inputs>0)] = 2
                    inputs[np.where(inputs<=0)] = -4
                elif BIM_config == 'target':
                    inputs = copy.deepcopy(clip_x[i:i+1, 0:n_his+1, :, :])
                    inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
                    temp = inputs[:,n_his,:,:]
                    temp[np.where(temp<=0)] = 2
                    temp[np.where(temp>0)] = -4
                    inputs[:, n_his, :, :] = temp
                else:
                    raise ValueError(f'ERROR: attacking type "{attacktype}" is not defined.')

                loss_gradient = sess.run(gradient, feed_dict={'data_input:0': inputs, 'keep_prob:0': 1.0})
                signed_grad = np.sign(loss_gradient)
                for j in range(n_his):
                    dif[i, :, :, :] = signed_grad[0,0,0,:, :, :]

            dif[np.where(dif>0)] = 1
            dif[np.where(dif<0)] = -1

            perturbations = np.zeros(np.shape(x_test))
            perturbations[:,0:n_his,:,:] = np.copy(dif)
            adversary = (BIM_para**0.5)*perturbations
            new_inputs = clip_x + adversary
            clip_x = clip(x_test, new_inputs, clip_e)

        adversary = clip_x - x_test
        print(f'the max value of the adversary is {np.max(adversary)}')
        print(f'the enery of the adversary is {np.sum(adversary[:,0:n_his,:,:]**2)/np.size(adversary[:,0:n_his,:,:])}')
        return adversary
