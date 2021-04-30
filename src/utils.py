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
    
def gather_nd(params, indices):
    nb_batch, _, nb_points = indices.shape
    res = torch.zeros(nb_batch, nb_points)
    for i in range(nb_batch):
        res[i] = params[i][indices[i][0], indices[i][1]]
    return res
    
def hungarian_huber_loss(x,y):
    # x shape : [batch_size, n_points, n_points]
    # y shape : [batch_size, n_points, n_points]
    
    huber_loss = torch.nn.MSELoss(reduction = 'none')
    pairwise_cost = torch.sum(huber_loss(torch.unsqueeze(x, axis = -3), torch.unsqueeze(y, axis = -2)), axis = -1)
    
    pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
    indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))

    actual_costs = gather_nd(pairwise_cost, indices)

    return torch.mean( torch.sum(actual_costs, axis = 1) )
    
    
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

