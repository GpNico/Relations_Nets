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
    """
        x shape : [batch_size, n_points, dim_points]
        y shape : [batch_size, n_points, dim_points]
    """

    loss = torch.nn.SmoothL1Loss(reduction = 'none')

    num_slot = x.shape[1]

    pairwise_cost = torch.mean(loss(torch.unsqueeze(x, axis = -3).expand(-1, num_slot, -1,-1),
                                    torch.unsqueeze(y, axis = -2).expand(-1, -1, num_slot,-1)), axis = -1)
    
    pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
    indices = torch.from_numpy(np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)), dtype = np.int64) ).cuda()
    
    sigmas = indices[:, 1]

    actual_costs = gather_nd(pairwise_cost, indices)

    return torch.mean( torch.sum(actual_costs, axis = 1) ), sigmas
    

def average_precision(pred, attributes, distance_threshold, data):
    """
         Args:
            pred: Array of shape [batch_size, num_elements, dimension] containing
              predictions. The last dimension is expected to be the confidence of the
              prediction.
            attributes: Array of shape [batch_size, num_elements, dimension] containing
              ground-truth object properties.
            distance_threshold: Threshold to accept match. -1 indicates no threshold.
        Returns:
            Average precision of the predictions.
    """

    def unsorted_id_to_image(detection_id, predicted_elements):
        return int(detection_id // predicted_elements)

    def process_targets(target, data):
        if data == 'multi_sprite':
            shape = np.argmax(target[:3])
            object_size = np.argmax(target[3:9])
            material = 1. #we do not predict material with multi_sprite
            color = np.argmax(target[9:16])
            coords = target[16:18]
            real_obj = target[18]
        elif data == 'clevr':
            shape = np.argmax(target[:3])
            object_size = np.argmax(target[3:5])
            material = np.argmax(target[5:7])
            color = np.argmax(target[7:15])
            coords = target[15:18]
            real_obj = target[18]
        elif data == 'multi_sprite_equal':
            shape = np.argmax(target[:5])
            object_size = np.argmax(target[5:10])
            material = 1. #we do not predict material with multi_sprite
            color = np.argmax(target[10:15])
            coords = target[15:17]
            real_obj = target[17]
            
        return coords, object_size, material, shape, color, real_obj

    batch_size, _, element_size = attributes.shape
    _, predicted_elements, _ = pred.shape

    flat_size = batch_size * predicted_elements
    flat_pred = np.reshape(pred, [flat_size, element_size])
    sort_idx = np.argsort(flat_pred[:, -1], axis = 0)[::-1] #Because last dim is for the confidence of  wheter or not there is an object

    sorted_predictions = np.take_along_axis(flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
    idx_sorted_to_unsorted = np.take_along_axis(np.arange(flat_size), sort_idx, axis=0)

    true_positives = np.zeros(sorted_predictions.shape[0])
    false_positives = np.zeros(sorted_predictions.shape[0])

    detection_set = set()

    for detection_id in range(sorted_predictions.shape[0]):

        current_pred = sorted_predictions[detection_id, :]

        original_image_idx = unsorted_id_to_image(idx_sorted_to_unsorted[detection_id], predicted_elements)

        gt_image = attributes[original_image_idx, :, :]

        best_distance = 10000
        best_id = None

        (pred_coords, pred_object_size, pred_material, pred_shape, pred_color, _) = process_targets(current_pred, data)

        for target_object_id in range(gt_image.shape[0]):
            target_object = gt_image[target_object_id, :]

            (target_coords, target_object_size, target_material, target_shape, target_color, target_real_obj) = process_targets(target_object, data)

            if target_real_obj:

                pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
                target_attr = [target_object_size, target_material, target_shape, target_color]

                match = pred_attr == target_attr

                if match:
                    # If a match was found, we check if the distance is below the
                    # specified threshold. Recall that we have rescaled the coordinates
                    # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
                    # `pred_coords`. To compare in the original scale, we thus need to
                    # multiply the distance values by 6 before applying the norm.
                    distance = np.linalg.norm((target_coords - pred_coords) * 1.)

                    if distance < best_distance:
                        best_distance = distance
                        best_id = target_object_id
        if best_distance < distance_threshold or distance_threshold == -1:
            if best_id is not None:
                if (original_image_idx, best_id) not in detection_set:
                    true_positives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))
                else:
                    false_positives[detection_id] = 1
            else:
                false_positives[detection_id] = 1
        else:
            false_positives[detection_id] = 1
    accumulated_fp = np.cumsum(false_positives)
    accumulated_tp = np.cumsum(true_positives)
    recall_array = accumulated_tp / np.sum(attributes[: , :, -1]) #divide by the number of objects
    precision_array = np.divide(accumulated_tp, (accumulated_tp + accumulated_fp))

    return compute_average_precision(np.array(precision_array, dtype=np.float32), np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [ i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i] ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  
  return average_precision

    
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

