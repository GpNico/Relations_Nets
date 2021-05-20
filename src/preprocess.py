# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import src.utils as utils
    
    
def ott_multi_sprite_all_carac(tensor, thresh = 0.5):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        confidence = 0.
        idx_sh = np.argmax(arr[k][:3])
        idx_si = np.argmax(arr[k][3:9])
        idx_co = np.argmax(arr[k][9:16])
        
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
    carac_coords = labels['coords']

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

            r, c = carac_coords[k][l]
            if r >= 0. and c >= 0.:
                ground_truth[k, l, 16] = r
                ground_truth[k, l, 17] = c
                ground_truth[k, l, 18] = 1 #There is an object
    
    return {'carac_labels': ground_truth.cuda()}
    
    
    
    
def ott_multi_sprite_all_carac(tensor, thresh = 0.5):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        confidence = 0.
        idx_sh = np.argmax(arr[k][:3])
        idx_si = np.argmax(arr[k][3:9])
        idx_co = np.argmax(arr[k][9:16])
        
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
    
    
def ggt_clevr_all_carac(labels, conf, pred):

    carac_shape = labels['shape']
    carac_scale = labels['size']
    carac_color = labels['color']
    carac_material = labels['material']
    carac_coords = labels['3d_coords']

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
            if scale == 0:
                ground_truth[k, l, 3] = 1
            elif scale == 1:
                ground_truth[k, l, 4] = 1
                
            scale = carac_material[k][l]
            if scale == 0:
                ground_truth[k, l, 5] = 1
            elif scale == 1:
                ground_truth[k, l, 6] = 1
                
            color = carac_color[k][l]
            if color == 0:
                ground_truth[k, l, 7] = 1
            elif color == 1:
                ground_truth[k, l, 8] = 1
            elif color == 2:
                ground_truth[k, l, 9] = 1
            elif color == 3:
                ground_truth[k, l, 10] = 1
            elif color == 4:
                ground_truth[k, l, 11] = 1
            elif color == 5:
                ground_truth[k, l, 12] = 1
            elif color == 6:
                ground_truth[k, l, 13] = 1
            elif color == 7:
                ground_truth[k, l, 14] = 1

            x, y, z = carac_coords[k][l]
            if x >= 0. and y >= 0. and z >= 0.:
                ground_truth[k, l, 15] = x
                ground_truth[k, l, 16] = y
                ground_truth[k, l, 17] = z
                
                ground_truth[k, l, 18] = 1 #There is an object
    
    return {'carac_labels': ground_truth.cuda()}
    
    
def ott_clevr_all_carac(tensor, thresh = 0.5):
    arr = tensor.cpu().detach().numpy()
    num_slots = arr.shape[0]
    
    title = ''
    
    for k in range(num_slots):
        confidence = 0.
        idx_sh = np.argmax(arr[k][:3])
        idx_si = np.argmax(arr[k][3:5])
        idx_ma = np.argmax(arr[k][5:7])
        idx_co = np.argmax(arr[k][7:15])
        
        confidence += arr[k][idx_sh] + arr[k][3 + idx_si] + arr[k][5 + idx_ma] + arr[k][7 + idx_co]  
        if confidence/4 < thresh:
            continue
            
        if idx_sh == 0:
            title += 'cube '
        elif idx_sh == 1:
            title += 'cylinder '
        elif idx_sh == 2:
            title += 'sphere '
            
        if idx_si == 0:
            title += 'small '
        elif idx_si == 1:
            title += 'large '
            
        if idx_ma == 0:
            title += 'metal '
        elif idx_ma == 1:
            title += 'rubber '
        
        if idx_co == 0:
            title += 'red '
        elif idx_co == 1:
            title += 'blue '
        elif idx_co == 2:
            title += 'purple '
        elif idx_co == 3:
            title += 'gray '
        elif idx_co == 4:
            title += 'cyan '
        elif idx_co == 5:
            title += 'brown '
        elif idx_co == 6:
            title += 'yellow '
        elif idx_co == 7:
            title += 'green '
            
        title += ' ; '
    
    return title
    
    
def gtt_multi_sprite_rela_base(labels, conf, pred):
    
    all_carac_labels = ggt_multi_sprite_all_carac(labels, conf, pred)['carac_labels']
    
    rela_label = labels['relation']
    
    num_slots, dim_rela = conf['params']['num_slots'], conf['prediction'][pred]['dim_rela']
    
    batch_size = rela_label.shape[0]
    
    def rela_to_Y(rela_code, num_slots = 4, dim_rela = 3):
        Y = torch.zeros(num_slots, num_slots, dim_rela)
        if rela_code == 1:
            Y[0,1,0] = 1
        elif rela_code == 2:
            Y[0,1,1] = 1
        elif rela_code == 7:
            Y[0,1,2] = 1
        return Y
        
    Y_full = []
    for k in range(batch_size):
        Y_full.append(rela_to_Y(rela_label[k], num_slots =  num_slots, dim_rela = dim_rela))
    Y_full = torch.stack(Y_full)
    #Y_full shape = [batch_size, n_slots, n_slots, dim_rela]
    
    dict_labels = {'carac_labels': all_carac_labels.cuda(), 'rela_labels': Y_full.cuda()}
    
    return dict_labels

