# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.optimize
import visdom
import os
import tqdm 
from importlib import import_module

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plotline(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
            
    def plotimagemask(self, var_name, imgs, masks, recons):
        seg_maps = visualize_masks(imgs, masks, recons)
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0], env=self.env)
        else:
            self.viz.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0], env = self.env, win = self.plots[var_name])
     
    def plotimage(self, var_name, img, label):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.image(img, env=self.env, opts={'caption': label})
        else:
            self.viz.image(img, env = self.env, win = self.plots[var_name], opts={'caption': label})


def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons):
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    return seg_maps
    
def to_class_label(y):
    """
        Take a one-hot-encoded vector and return the class number
    """
    
    batch_size , num_classes = y.shape 
    
    target = torch.zeros(batch_size)
    
    for k in range(batch_size):
        class_num = torch.argmax(y)
        if y[class_num] ==1: #Class 0 correspond to "no object"
            target[k] = class_num + 1
        
    return target.type(torch.LongTensor)
    
def gather_nd(params, indices):
    nb_batch, _, nb_points = indices.shape
    res = torch.zeros(nb_batch, nb_points)
    for i in range(nb_batch):
        res[i] = params[i][indices[i][0], indices[i][1]]
    return res
    
def huber_loss(x,y, delta = 1.):
    
    L1 = torch.nn.L1Loss(reduction = 'none')(x,y)
    L2 = torch.nn.MSELoss(reduction = 'none')(x, y)
    
    val = torch.where(L1 < delta, 0.5*L2, delta*(L1 - 0.5*delta))
    
    return val
      
def hungarian_huber_loss(x,y):
    """
        x shape : [batch_size, n_points, dim_points]
        y shape : [batch_size, n_points, dim_points]
    """
    #loss = torch.nn.MSELoss(reduction = 'none')
    loss = huber_loss
    pairwise_cost = torch.sum(loss(torch.unsqueeze(x, axis = -3), torch.unsqueeze(y, axis = -2)), axis = -1)
    
    pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
    indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))

    actual_costs = gather_nd(pairwise_cost, indices)

    return torch.mean( torch.sum(actual_costs, axis = 1) )
    
    
def hybrid_hungarian_huber_ce_loss(x, y):
    loss_1 = huber_loss
    pairwise_cost = torch.sum(loss_1(torch.unsqueeze(x, axis = -3), torch.unsqueeze(y, axis = -2)), axis = -1)
    
    pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
    _, sigma = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))
    
    # \tilde{o}_{\sigma(i)} \sim o_i
    # rq: \tilde{o}_i (prediction) ; o_i (ground truth)
    
    loss_2 = torch.nn.CrossEntropyLoss(reduction = 'none')
    
    batch_size, num_slots, _ = x.shape #Not Clean
    loss = torch.zeros(batch_size)
    
    for k in range(num_slots):
        loss += loss_2(x[:, sigma[k], 0:3], to_class_label(y[k, 0:3]))
        loss += loss_2(x[:, sigma[k], 3:9], to_class_label(y[k, 3:9]))
        loss += loss_2(x[:, sigma[k], 9:16], to_class_label(y[k, 9:16]))
        
    return torch.mean(loss)
    
    
    
def Average_Precision(model, loader, conf, thresh = 0.01):
    N = len(loader)
    model.eval()
    precision = 0.
    
    huber_loss = torch.nn.MSELoss(reduction = 'none')
    
    for data in tqdm.tqdm(loader, total = N, disable = False):
        images, labels = data
        ground_truth = get_ground_truth(labels, conf).cuda()
        
        images = images.cuda()
        output, _ = model(images)
        
        x = output.detach()
        y = ground_truth
        batch_size = ground_truth.shape[0]
        
        pairwise_cost = torch.sum(huber_loss(torch.unsqueeze(x, axis = -3), torch.unsqueeze(y, axis = -2)), axis = -1)
    
        pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
        indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))

        actual_costs = gather_nd(pairwise_cost, indices)
        
        for k in range(batch_size):
            for l in range(num_slots):
                if actual_costs[k][l] < thresh:
                    precision += 1./(conf.num_slots*batch_size)
        
    return precision/N
    
def import_from_path(path_to_module, obj_name = None):
    """
    Import an object from a module based on the filepath of
    the module and the string name of the object.
    If obj_name is None, return the module instead.
    """
    module_name = path_to_module.replace("/",".").strip(".py")
    module = import_module(module_name)
    if obj_name == None:
        return module
    obj = getattr(module, obj_name)
    return obj
    
def color_name(arr):
    assert arr.shape[0] == 3, "Color array must be of shapr 3"
    
    x, y, z = arr
    
    if x == 255:
        if y == 0 and z == 0:
            return 'red'
        elif y == 255 and z == 0:
            return 'yellow'
        elif y == 255 and z == 255:
            return 'white'
        elif y == 0 and z == 255:
            return 'magenta'
    elif x == 0:
        if y == 255 and z == 0:
            return 'green'
        elif y == 255 and z == 255:
            return 'cyan'
        elif y == 0 and z == 255:
            return 'blue'

