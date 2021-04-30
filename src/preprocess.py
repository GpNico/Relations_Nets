# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

 
def ott_multi_sprite_rela_base(tensor, thresh = 0.5):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        idx = np.argmax(arr[k])
        if arr[k][idx] < thresh:
            continue
        if idx == 0:
            title += 'top '
            #title += 'square '
        elif idx == 1:
            title += 'right '
            #title += 'triangle '
        elif idx == 2:
            title += 'right_and_top '
            #title += 'ellipse '
    
    return title
    
    
def ggt_multi_sprite_rela_base(labels, conf, pred):

    carac = labels['relation'].type(torch.LongTensor).unsqueeze(dim = -1)

    batch_size, max_objs = carac.shape
    
    ground_truth = torch.zeros(batch_size, conf['params']['num_slots'], conf['prediction'][pred]['dim_points'])
    
    for k in range(batch_size):
        for l in range(max_objs):
            val = carac[k][l]
            if val == 1:
                ground_truth[k, l, 0] = 1
            elif val == 2:
                ground_truth[k, l, 1] = 1
            elif val == 7:
                ground_truth[k, l, 2] = 1
    
    return ground_truth
    
    
def ott_multi_sprite_shape(tensor, thresh = 0.5):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        idx = np.argmax(arr[k])
        if arr[k][idx] < thresh:
            continue
        if idx == 0:
            title += 'square '
        elif idx == 1:
            title += 'triangle '
        elif idx == 2:
            title += 'ellipse '
    
    return title
    
def ggt_multi_sprite_shape(labels, conf, pred):

    carac = labels['shape'].type(torch.LongTensor)#.unsqueeze(dim = -1)

    batch_size, max_objs = carac.shape
    
    ground_truth = torch.zeros(batch_size, conf['params']['num_slots'], conf['prediction'][pred]['dim_points'])
    
    for k in range(batch_size):
        for l in range(max_objs):
            val = carac[k][l]
            if val == 0:
                ground_truth[k, l, 0] = 1
            elif val == 1:
                ground_truth[k, l, 1] = 1
            elif val == 2:
                ground_truth[k, l, 2] = 1
    
    return ground_truth

