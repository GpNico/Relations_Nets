# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import src.utils as utils
    
    
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

def ggt_multi_sprite_equal_carac(labels, conf, pred):

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
            elif shape == 3:
                ground_truth[k, l, 3] = 1
            elif shape == 4:
                ground_truth[k, l, 4] = 1
                
            scale = carac_scale[k][l]
            if scale == 0.5000:
                ground_truth[k, l, 5] = 1
            elif scale == 0.6250:
                ground_truth[k, l, 6] = 1
            elif scale == 0.7500:
                ground_truth[k, l, 7] = 1
            elif scale == 0.8750:
                ground_truth[k, l, 8] = 1
            elif scale == 1.:
                ground_truth[k, l, 9] = 1
                
            color = utils.color_name(carac_color[k][l])
            if color == 'red':
                ground_truth[k, l, 10] = 1
            elif color == 'yellow':
                ground_truth[k, l, 11] = 1
            elif color == 'white':
                ground_truth[k, l, 12] = 1
            elif color == 'green':
                ground_truth[k, l, 13] = 1
            elif color == 'blue':
                ground_truth[k, l, 14] = 1

            r, c = carac_coords[k][l]
            if r >= 0. and c >= 0.:
                ground_truth[k, l, 15] = r
                ground_truth[k, l, 16] = c
                ground_truth[k, l, 17] = 1 #There is an object
    
    return {'carac_labels': ground_truth.cuda()}
    
    
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
                
            material = carac_material[k][l]
            if material == 0:
                ground_truth[k, l, 5] = 1
            elif material == 1:
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
        elif rela_code == 8:
            Y[0,1,2] = 1
        return Y
        
    Y_full = []
    for k in range(batch_size):
        Y_full.append(rela_to_Y(rela_label[k], num_slots =  num_slots, dim_rela = dim_rela))
    Y_full = torch.stack(Y_full)
    #Y_full shape = [batch_size, n_slots, n_slots, dim_rela]
    
    dict_labels = {'carac_labels': all_carac_labels.cuda(), 'rela_labels': Y_full.cuda()}
    
    return dict_labels


def gtt_multi_sprite_rela_contact(labels, conf, pred):
    
    all_carac_labels = ggt_multi_sprite_all_carac(labels, conf, pred)['carac_labels']
    
    rela_label = labels['relation']
    
    num_slots, dim_rela = conf['params']['num_slots'], conf['prediction'][pred]['dim_rela']
    
    batch_size = rela_label.shape[0]
    
    def rela_to_Y(rela_code, num_slots = 4, dim_rela = 3):
        Y = torch.zeros(num_slots, num_slots, dim_rela)
        if rela_code == 0:
            Y[0,1,0] = 1
        elif rela_code == 1:
            Y[0,1,1] = 1
        elif rela_code == 2:
            Y[0,1,2] = 1
        elif rela_code == 3:
            Y[0,1,3] = 1
        elif rela_code == 4:
            Y[0,1,4] = 1
        elif rela_code == 5:
            Y[0,1,5] = 1
        elif rela_code == 6:
            Y[0,1,6] = 1
        elif rela_code == 7:
            Y[0,1,7] = 1
        return Y
        
    Y_full = []
    for k in range(batch_size):
        Y_full.append(rela_to_Y(rela_label[k], num_slots =  num_slots, dim_rela = dim_rela))
    Y_full = torch.stack(Y_full)
    #Y_full shape = [batch_size, n_slots, n_slots, dim_rela]
    
    dict_labels = {'carac_labels': all_carac_labels.cuda(), 'rela_labels': Y_full.cuda()}
    
    return dict_labels
