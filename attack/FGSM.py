from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from os.path import join as pjoin

import tensorflow as tf
import numpy as np

import copy

def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def attack_test(inputs, clean_inputs, x_stats, load_path, batch_size, n_his, n_pred):
    '''
    attacking test function
    :param inputs: data with noise
    :param clean_inputs: data without noise
    :param x_stats: statistical information of data
    :param load_path: path of the graph and model
    :param batch_size: batch size
    :param n_his: length of historical data
    :param n_pred: time steps of prediction
    '''
    # load model
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as test_sess:

        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        step_idx = tmp_idx = np.arange(3, n_pred+1, 3) - 1 

        # prediction
        y_test, len_test = multi_pred(test_sess, pred, inputs, batch_size, n_his, n_pred, step_idx)
#        np.save('./output/ypred/ytest',y_test)
        # evaluation
        evl = evaluation(clean_inputs[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')



def generate_adversary(load_path,x_test, n_his, n_pred, FGSM_para, FGSM_config):
    '''
    function to generate adversary based on FGSM
    :param load_path: path of the graph and model
    :param x_test: test data
    :param n_his: length of historical data
    :param n_pred: time steps of prediction
    :param FGSM_para: l-infinite of the adversary
    :param FGSM_config: types of method ["none","back","target","target2"]
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
        dif = np.zeros((np.size(x_test, 0), n_his, np.size(x_test,2), np.size(x_test,3)))
        
        for i in range(np.size(x_test, 0)-n_frame):

            if FGSM_config == 'none':
                inputs = copy.deepcopy(x_test[i:i+1, 0:n_his+1, :, :])
            elif FGSM_config == 'back':
                inputs = copy.deepcopy(x_test[i:i+1, 0:n_his+1, :, :])
                inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
            elif FGSM_config == 'target2':
                inputs = copy.deepcopy(x_test[i:i+1, 0:n_his+1, :, :])
                inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
                inputs[np.where(inputs>0)] = 2
                inputs[np.where(inputs<=0)] = -4
            elif FGSM_config == 'target':
                inputs = copy.deepcopy(x_test[i:i+1, 0:n_his+1, :, :])
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
#                dif[i+j, j, :, :] = signed_grad[0,0,0,j, :, :]
                dif[i, :, :, :] = signed_grad[0,0,0,:, :, :]

#        diff = dif.sum(axis=1)
        diff = dif
        diff[np.where(diff>0)] = 1
        diff[np.where(diff<0)] = -1

        perturbations = np.zeros(np.shape(x_test))
        perturbations[:,0:n_his,:,:] = np.copy(diff)
#        for i in range(np.size(x_test,0)-n_frame):
#            perturbations[i, :, :, :] = diff[i:i+n_frame,:,:]

#        adversary = x_test + (FGSM_para**0.5)*perturbations
        adversary = (FGSM_para**0.5)*perturbations
        print(f'the max value of the adversary is:{np.max(adversary)}')
        print(f'the energy of the adversary is:{np.sum(adversary[:,0:n_his,:,:]**2)/np.size(adversary[:,0:n_his,:,:])}')
        return adversary
