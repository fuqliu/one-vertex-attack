import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os.path import join as pjoin
from data_loader.data_utils import *
from utils.math_utils import evaluation

import tensorflow as tf
import numpy as np
import argparse
import time
from attack.FGSM import *
from attack.iterative import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--load_path', type=str, default='./output/models/')
parser.add_argument('--attack_type', type=str, default='FGSM')
parser.add_argument('--noise_energy', type=float, default=0.1)
parser.add_argument('--FGSM_para', type=float, default=0.001)
parser.add_argument('--FGSM_config', type=str, default='target')
parser.add_argument('--BIM_config', type=str, default='target')
parser.add_argument('--BIM_para', type=float, default=0.000025)
parser.add_argument('--clip_epoch', type=int, default=20)
parser.add_argument('--clip_e', type=float, default=0.1)

args = parser.parse_args()
print(f'Training configs: {args}')

attacktype = ['none','GWN','FGSM','BIM']
n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
n_train, n_val, n_test = 34, 5, 5
load_path = args.load_path
batch_size = args.batch_size
attack_type = args.attack_type
noise_energy = args.noise_energy
FGSM_config, FGSM_para = args.FGSM_config, args.FGSM_para
BIM_config, BIM_para = args.BIM_config, args.BIM_para
clip_epoch, clip_e = args.clip_epoch, args.clip_e

# data loading
print('===================data loading===================')
data_file = f'PeMSD7_V_{n}.csv'
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
x_train = PeMS.get_data('train')
x_test, x_stats = PeMS.get_data('test'), PeMS.get_stats()
print(f'data loaded from:', './dataset'+data_file)

# attacking
print('===================adversary generating===================')
print('generating adversarial samples, the attack method is: '+attack_type)
if attack_type == attacktype[0]:

    adversarial_x_test = x_test

elif attack_type == attacktype[1]:

    noise_mean = 0.0
    noise = np.random.normal(noise_mean, noise_energy**0.5, size = np.size(x_test))
    noise = np.reshape(noise, x_test.shape)
    print(f'the max value of the GWN is: {np.max(noise)}')
    print(f'the energy of the GWN is: {np.sum(noise**2)/np.size(noise)}')
    adversarial_x_test = x_test + noise

elif attack_type == attacktype[2]:
    
    noise = generate_adversary(load_path, x_test, n_his, n_pred, FGSM_para, FGSM_config)
    adversarial_x_test = x_test + noise

elif attack_type == attacktype[3]:
    
    noise = ite_generate_adversary(load_path, x_test, n_his, n_pred, BIM_para, BIM_config, clip_epoch, clip_e)
    adversarial_x_test = x_test + noise

else:
    raise ValueError(f'ERROR: attacking type "{attack_type}" is not defined.')

start_time = time.time()

#np.save('./output/ypred/adversary',adversarial_x_test)
# attacking test
print('===================attacking test===================')

attack_test(adversarial_x_test, x_test, x_stats, load_path, batch_size, n_his, n_pred)

print(f'Model Test Time {time.time() - start_time:.3f}s')
print('Testing model finished!')

