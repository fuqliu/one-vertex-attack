from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from models.layers import *
from os.path import join as pjoin

import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
import time
import numpy as np

def get_variables_to_restore(variables, var_keep_dic):

    variables_to_restore = []
    for v in variables:
        # one can do include or exclude operations here.
        if v.name.split(':')[0] in var_keep_dic:
            print("Variables restored: %s" % v.name)
            variables_to_restore.append(v)

    return variables_to_restore

def get_variable_in_checkpoint_file(pretrained_model):

    var_keep_dic = []
    reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model)
    param_dict = reader.get_variable_to_shape_map()

    for key, val in param_dict.items():
        var_keep_dic.append(key)

    return var_keep_dic
