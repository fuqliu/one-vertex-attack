import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os.path import join as pjoin
from data_loader.data_utils import *
from utils.math_graph import *

import tensorflow as tf
import numpy as np
import argparse
import time
from attack.universal import *
from attack.onevertex import *
import attack.feed_onevertex_2 as feed_onevertex
from attack.FGSM import attack_test as FGSM_attack_test

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--save', type=int, default=100)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-1)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--theta', type=float, default=1000)
parser.add_argument('--noise_para', type=float, default=0.01)
parser.add_argument('--generator_channel', type=int, default=5)
parser.add_argument('--flag', type=str, default='feed')
parser.add_argument('--position', type=int, default=54)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--val', type=int, default=0)

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
theta = args.theta
generator_channel = args.generator_channel
train_flag = args.flag
batch_size = args.batch_size
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = f'PeMSD7_V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')


load_path = "./output/models/"

if train_flag == 'universal':

    universal_train(PeMS, blocks, args, load_path)
    universal_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)

elif train_flag == 'onevertex':

    onevertex_generator_train(PeMS, blocks, args, load_path)
    onevertex_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)

elif train_flag == 'feed':

    if args.position >= n | args.position < 0:
        
        print(f'the positon should between 0 and {n}')
    
    else:
        feed_onevertex.feed_onevertex_train(PeMS, blocks, args, load_path)
        # test
#        x_test, x_stats = PeMS.get_data('test'), PeMS.get_stats()
#        adversarial_x_test = x_test + trained_noise
#        FGSM_attack_test(adversarial_x_test, x_test, x_stats, load_path, batch_size, n_his, n_pred)

else:

    print(f'training flag: {train_flag} is invalid')

