import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os.path import join as pjoin
from data_loader.data_utils import *
from utils.math_graph import *
import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils.math_utils import evaluation


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


def onevertex_attack_test(inputs, x_stats, load_path, batch_size, n_his, n_pred, adversary):
    '''
    attacking test function
    :param inputs: data with noise
    :param x_stats: statistical information of data
    :param load_path: path of the graph and model
    :param batch_size: batch size
    :param n_his: length of historical data
    :param n_pred: time steps of prediction
    :param adversary: perturbations
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
        
        # test the prediction result with one vertex noise
        attention_matrix = np.zeros([1,np.size(x_train,2)])
        for i in range(np.size(x_train,2)):
            x_test = np.copy(inputs)
            x_test[:,0:n_his,i,:] = x_test[:,0:n_his,i,:]+ adversary[:,0:n_his,i,:]
        # prediction
            y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
#        np.save('./output/ypred/ytest',y_test)
        # evaluation
            evl = evaluation(inputs[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
            for ix in tmp_idx:
                te = evl[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
            attention_matrix[0,i] = te[0]
    return attention_matrix

def clip(inputs, new_inputs, clip_e):
    temp = np.zeros((np.size(inputs,0), np.size(inputs,1), np.size(inputs,2), np.size(inputs,3), 2))
    temp[:,:,:,:,0] = inputs + clip_e
    temp[:,:,:,:,1] = np.max([new_inputs,inputs-clip_e],0)
    clip_x = np.min(temp,4)
    return clip_x

def generate_adversary_everyvertex(load_path, x_train, n_his, n_pred, BIM_para, clip_epoch, clip_e):
    '''
    function to generate adversary based on the Basic Iterative Method
    :param load_path: path of the graph and model
    :param x_train: test data
    :param n_his: length of historical data
    :param n_pred: time steps of prediction
    :param BIM_para: l-infinite of the adversary
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
                
        clip_x = np.copy(x_train)
        
        for epoch in range(clip_epoch):
            dif = np.zeros((np.size(x_train, 0), n_his, np.size(x_train,2), np.size(x_train,3)))
            
            for i in range(np.size(x_train, 0)-n_frame):

                inputs = copy.deepcopy(clip_x[i:i+1, 0:n_his+1, :, :])
                inputs[:, n_his, :, :] = inputs[:, n_his-1, :, :]
                temp = inputs[:,n_his,:,:]
                temp[np.where(temp<=0)] = 2
                temp[np.where(temp>0)] = -4
                inputs[:, n_his, :, :] = temp

                loss_gradient = sess.run(gradient, feed_dict={'data_input:0': inputs, 'keep_prob:0': 1.0})
                signed_grad = np.sign(loss_gradient)
                for j in range(n_his):
                    dif[i, :, :, :] = signed_grad[0,0,0,:, :, :]
            print(f'=====the current epoch is :{epoch}=====')
            dif[np.where(dif>0)] = 1
            dif[np.where(dif<0)] = -1

            perturbations = np.zeros(np.shape(x_train))
            perturbations[:,0:n_his,:,:] = np.copy(dif)
            adversary = (BIM_para**0.5)*perturbations
            new_inputs = clip_x + adversary
            clip_x = clip(x_train, new_inputs, clip_e)

        adversary = clip_x - x_train
        print(f'the max value of the adversary is {np.max(adversary)}')
        print(f'the enery of the adversary is {np.sum(adversary[:,0:n_his,:,:]**2)/np.size(adversary[:,0:n_his,:,:])}')
        return adversary

#paramaters
load_path = "./output/models/"
n_his = 12
n_pred = 9
para = 0.00025 
clip_epoch = 10
clip_e = 0.5
batch_size = 100
n = 228
# Data Preprocessing
data_file = f'PeMSD7_V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
# get data
x_train, x_stats = PeMS.get_data('train'), PeMS.get_stats()
# generate adversary
adversary = generate_adversary_everyvertex(load_path, x_train, n_his, n_pred, para, clip_epoch, clip_e)
np.save('./output/onevertex/adversary',adversary)

attention_matrix = onevertex_attack_test(x_train, x_stats, load_path, batch_size, n_his, n_pred, adversary)
plt.figure(figsize=(16,8))
plt.plot(np.size(np.size(attention_matrix)),attention_matrix,color='#6140ef',linewidth=5,label="attaention")
plt.tick_params(labelsize=23,direction="in",width=3,length=9)
    
bwith = 3
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
    
plt.show()
