import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin
from attack.de import attacking_map
import tensorflow as tf
import numpy as np
from data_loader.data_utils import *
from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
import time
from attack.de import *
import matplotlib.pyplot as plt

n = 228
n_his = 12
n_pred = 9
run_flag = 'run'

if run_flag == 'run':

    # Data Preprocessing
    data_file = f'PeMSD7_V_{n}.csv'
    n_train, n_val, n_test = 34, 5, 5
    PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
    print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
    x_test, x_stats = PeMS.get_data('test'), PeMS.get_stats()
    attacking_map(PeMS, n, n_his, n_pred)

    # attacking map
elif run_flag == 'map':

    candidates = np.load('./output/ypred/candidates_attacking_map.npy')
    attacking_map = np.zeros([n,2])
    attacking_map[:,0] = [i for i in range(n)]
    for index in range(n):
        temp = candidates[:,0]
        attacking_map[index,1] = np.size(temp[np.where(temp==index)])
    np.save('./output/attacking_map',attacking_map)
    print(attacking_map)
else:
    attacking_map = np.load('./output/attacking_map.npy')
    plt.figure(figsize=(16,8))
    e = np.linspace(0,n,n)
    plt.plot(e,attacking_map,color='#fc5a50',linewidth=5,label="None",marker="o",markersize=10)
    plt.show()