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
'''
def build_onevertex_noise_model(inputs, n, n_his, Ks, Kt, blocks, keep_prob, theta, noise_para, position):

    x = inputs[:, 0:n_his, :, :]
    # define one vertex adversary
    one_vertex_noise = tf.compat.v1.get_variable(name='one_noise', shape=[1,n_his, n, 1], dtype=tf.float32)
    tf.add_to_collection(name='one_noise', value=one_vertex_noise)

    # add noise
#    mark_ = np.zeros([1,n_his,n,1])
#    mark_[:,:,position:position+1,:] = np.ones([1,n_his,1,1])
#    mark_ = mark_.tolist()
#    mark_ = tf.constant(mark_,dtype=tf.float32)
#    x = x + mark_*one_vertex_noise
    x = x + tf.tile(one_vertex_noise,[tf.shape(x)[0],1,1,1])
#    x = x + one_vertex_noise
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
    target = 3*tf.sign(inputs[:, n_his:n_his + 1, :, :])-0.999999999999999
    # define loss
    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='target_loss',
                         value=tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='noise_loss',
                         value=tf.math.reduce_sum(tf.math.pow(one_vertex_noise[:,:,position,:],2))/(n_his))

#    train_loss = tf.nn.l2_loss(y - target) + theta*noise_l2_loss(one_vertex_noise[:,:,position,:],noise_para,n_his)
    train_loss = tf.nn.l2_loss(y - target) 
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)

    gradient = tf.gradients(train_loss, one_vertex_noise)
    gradient = tf.convert_to_tensor(gradient,dtype=tf.float32)
    tf.add_to_collection(name='gradient',value=gradient)
    return train_loss, single_pred


def feed_onevertex_train(inputs, blocks, args, load_path, sum_path='./output/onevertex/feed'):

#    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

#    test_graph = tf.Graph()

#    with test_graph.as_default():
#        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    theta = args.theta
    noise_para = args.noise_para
    position = args.position

    # Placeholder for model training
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')


    # Define model loss
    train_loss, pred = build_onevertex_noise_model(x, n, n_his, Ks, Kt, blocks, keep_prob, theta, noise_para, position)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)
    target_loss = tf.add_n(tf.get_collection('target_loss'))
    tf.summary.scalar('target_loss', target_loss)
    noise_loss = tf.add_n(tf.get_collection('noise_loss'))
    tf.summary.scalar('noise_loss', noise_loss)
    one_noise = tf.get_collection('one_noise')
    gradient = tf.get_collection('gradient')

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
#    len_train = inputs.get_len('train')
#    if len_train % batch_size == 0:
#        epoch_step = len_train / batch_size
#    else:
#        epoch_step = int(len_train / batch_size) + 1

    # Learning rate decay with rate 0.7 every 5 epochs.
#    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=10, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)

    # freeze variables
    trainable_vars = tf.trainable_variables()
    onevertex_noise_var_list = [t for t in trainable_vars if t.name.startswith(u'one_noise')]
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(loss = train_loss, var_list = onevertex_noise_var_list)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss, var_list = onevertex_noise_var_list)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    # data prepare
    x_test = inputs.get_data('test')

    # initialize the trained_noise and predication
    trained_noise = np.zeros(np.shape(x_test))

    #get the pretrained model
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    # begin training the one vertex noise
    with tf.Session() as sess:

        # initialize all variables
        variables = tf.global_variables()
        sess.run(tf.global_variables_initializer())

        # get the trained variables
        var_keep_dic = get_variable_in_checkpoint_file(model_path)
        variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, model_path)
        print("======================Pre-trained model has been loaded======================.")

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

        # train one vertex noise one by one
        for index in range(np.size(x_test, 0)):
            start_time = time.time()
            for i in range(epoch):
                inputs_data = np.tile(x_test[index:index+1, 0:n_his + 1, :, :],[50,1,1,1])
                summary, _ = sess.run([merged, train_op], feed_dict={'data_input:0': inputs_data, 'keep_prob:0': 1.0})
                loss_value = sess.run([train_loss, copy_loss, target_loss, noise_loss], feed_dict={'data_input:0': inputs_data, 'keep_prob:0': 1.0})
                print(f'index {index:2d}, Epoch {i:2d}: train_loss-{loss_value[0]:.3f}, copy_loss-{loss_value[1]:.3f}, target_loss-{loss_value[2]:.3f}, noise_loss-{loss_value[3]:.3f}')
                gradient_ = sess.run(gradient,feed_dict={'data_input:0': inputs_data, 'keep_prob:0': 1.0})
                gradient_ = np.array(gradient_)
                noise_ = np.array(sess.run(one_noise))
                print(noise_[0,0,:,position,0])
            print(f'index {index:2d} Training Time {time.time() - start_time:.3f}s')
            # collect the noise
            noise_ = np.array(sess.run(one_noise))
            trained_noise[index,0:n_his,position,:] = noise_[0,0,:,position,:]
    print('Training model finished!')
    return trained_noise
'''

def build_onevertex_noise_model(inputs, n_his, Ks, Kt, blocks, keep_prob, theta, batch_size, n, noise_para, position):
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
    universal_noise = tf.compat.v1.get_variable(name='onenoise', shape=[1,n_his, n, 1], dtype=tf.float32)
    tf.add_to_collection(name='one_noise', value=universal_noise)
    # add noise
    mark_ = np.zeros([1,n_his,n,1])
    mark_[:,:,position,:] = 1
    mark_ = mark_.tolist()
    mark = tf.constant(mark_,dtype=tf.float32) 
    x = x + tf.tile(mark*universal_noise,[tf.shape(x)[0],1,1,1])
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
                         value=2*tf.nn.l2_loss(universal_noise[:,:,position,:]))
    train_loss = tf.nn.l2_loss(y - target) + theta*noise_l2_loss(universal_noise[:,:,position,:],noise_para,n_his)
#    train_loss = tf.nn.l2_loss(y - target) 
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def feed_onevertex_train(inputs, blocks, args, load_path, sum_path='./output/onevertex/feed'):
# build up a new graph, and create a session as: sess
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    theta = args.theta
    noise_para = args.noise_para
    position = args.position
    index = args.index
    # Placeholder for model training
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    # get the check point of the pretrained model
    model_path = tf.train.get_checkpoint_state(load_path)
    pretrained_model = model_path.model_checkpoint_path

    # Define model loss
    train_loss, pred = build_onevertex_noise_model(x, n_his, Ks, Kt, blocks, keep_prob, theta, batch_size, n, noise_para, position)
    
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)
    target_loss = tf.add_n(tf.get_collection('target_loss'))
    tf.summary.scalar('target_loss', target_loss)
    noise_loss = tf.add_n(tf.get_collection('noise_loss'))
    tf.summary.scalar('noise_loss', noise_loss)
    onenoise = tf.get_collection('one_noise')
    
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
#    len_train = inputs.get_len('train')
#    if len_train % batch_size == 0:
#        epoch_step = len_train / batch_size
#    else:
#        epoch_step = int(len_train / batch_size) + 1
    epoch_step = 30
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps-9150, decay_steps=epoch_step, decay_rate=0.7, staircase=True)
#    lr = args.lr
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    # freeze variables
    trainable_vars = tf.trainable_variables()
    universal_noise_var_list = [t for t in trainable_vars if t.name.startswith(u'onenoise')]
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(loss = train_loss, var_list = universal_noise_var_list)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss, var_list = universal_noise_var_list)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()
    
    #GPU memory setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config=tf.ConfigProto(gpu_options=gpu_options)
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
#        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
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

        x_test = inputs.get_data('test')
        inputs_data = np.tile(x_test[index,:,:,:],[50,1,1,1])
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs_data, batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
#                writer.add_summary(summary, i * epoch_step + j)
                loss_value = sess.run([train_loss, copy_loss, target_loss, noise_loss], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                print(f'Epoch {i:2d}, Step {j:3d}: train_loss-{loss_value[0]:.3f}, copy_loss-{loss_value[1]:.3f}, target_loss-{loss_value[2]:.3f}, noise_loss-{loss_value[3]:.3f}')
                print('the current global_steps is: %s:' % sess.run(global_steps))

            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')
        temp = sess.run(onenoise)
        print(np.shape(np.array(temp)))
        #save the noise
        np.save(pjoin(f'./output/onevertex/feed/result/{position}_{index}_{args.val}'),np.array(temp)[0,0,:,position,0])
        print(f'noise-{position}_{index} saved, copy_loss-{loss_value[1]:.3f}, target_loss-{loss_value[2]:.3f}')
        file = open(f'./output/onevertex/feed/result/{position}_{args.val}log.txt','a')
        file.write(f'{position}_{index}:copy_loss-{loss_value[1]:.3f}, target_loss-{loss_value[2]:.3f} \n')
        file.close()
#            if (i + 1) % args.save == 0:
#                model_save(sess, global_steps, 'universal', './output/universalattacking/models/')
#        writer.close()
    print(f'========Training model finished!:{index}=========')
