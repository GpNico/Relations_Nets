# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import src.utils as utils

 
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
    
    
def ott_multi_sprite_all_carac(tensor, thresh = 0.1):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        confidence = 0.
        idx_sh = np.argmax(arr[k][:3])
        idx_si = np.argmax(arr[k][3:9])
        idx_co = np.argmax(arr[k][9:])
        
        confidence += arr[k][idx_sh] + arr[k][3 + idx_si] + arr[k][9 + idx_co]  
        if confidence/3 < thresh:
            continue
            
        if idx_sh == 0:
            title += 'square '
        elif idx_sh == 1:
            title += 'triangle '
        elif idx_sh == 2:
            title += 'ellipse '
            
        if idx_si == 0:
            title += '0.5 '
        elif idx_si == 1:
            title += '0.6 '
        elif idx_si == 2:
            title += '0.7 '
        elif idx_si == 3:
            title += '0.8 '
        elif idx_si == 4:
            title += '0.9  '
        elif idx_si == 5:
            title += '1. '
        
        if idx_co == 0:
            title += 'red '
        elif idx_co == 1:
            title += 'yellow '
        elif idx_co == 2:
            title += 'white '
        elif idx_co == 3:
            title += 'magenta '
        elif idx_co == 4:
            title += 'green '
        elif idx_co == 5:
            title += 'cyan '
        elif idx_co == 6:
            title += 'blue '
            
        title += ' ; '
    
    return title
    
    
def ggt_multi_sprite_all_carac(labels, conf, pred):

    carac_shape = labels['shape']
    carac_scale = labels['scale']
    carac_color = labels['color']

    batch_size, max_objs = carac_shape.shape
    
    ground_truth = torch.zeros(batch_size, conf['params']['num_slots'], conf['prediction'][pred]['dim_points'])
    
    for k in range(batch_size):
        for l in range(max_objs):
            shape = carac_shape[k][l]
            if shape == 0:
                ground_truth[k, l, 0] = 1
            elif shape == 1:
                ground_truth[k, l, 1] = 1
            elif shape == 2:
                ground_truth[k, l, 2] = 1
                
            scale = carac_scale[k][l]
            if scale == 0.5:
                ground_truth[k, l, 3] = 1
            elif scale == 0.6:
                ground_truth[k, l, 4] = 1
            elif scale == 0.7:
                ground_truth[k, l, 5] = 1
            elif scale == 0.8:
                ground_truth[k, l, 6] = 1
            elif scale == 0.9:
                ground_truth[k, l, 7] = 1
            elif scale == 1.:
                ground_truth[k, l, 8] = 1
                
            color = utils.color_name(carac_color[k][l])
            if color == 'red':
                ground_truth[k, l, 9] = 1
            elif color == 'yellow':
                ground_truth[k, l, 10] = 1
            elif color == 'white':
                ground_truth[k, l, 11] = 1
            elif color == 'magenta':
                ground_truth[k, l, 12] = 1
            elif color == 'green':
                ground_truth[k, l, 13] = 1
            elif color == 'cyan':
                ground_truth[k, l, 14] = 1
            elif color == 'blue':
                ground_truth[k, l, 15] = 1
    
    return ground_truth

