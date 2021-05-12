# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import scipy.optimize


class TrainingMonitor:

    def __init__(self, dataset = 'clevr'):
        
        self.dataset = dataset


    def process_targets(self, target, data):
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
            
        return coords, object_size, material, shape, color, real_obj

    def get_carac_precision(self, preds, targets):
        """
        Args:
            preds : tensor [batch_size, num_slots, dim_points]
            targets : tensor [batch_size, num_slots, dim_points]
        Returns:
            precision : float
        """

        color_count = 0
        shape_count = 0
        size_count = 0

        num_objs = 0

        sigmas = self.get_alignement_indices(preds, targets)

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        
        batch_size, num_slots = sigmas.shape

        for k in range(batch_size):
            for l in range(num_slots):
                if targets[k][l].max() > 0.: #There is an object in this slot

                    num_objs += 1

                    (_, pred_object_size, pred_material, pred_shape, pred_color, _) = self.process_targets(preds[k][sigmas[k][l]], self.dataset)
                    (_, target_object_size, target_material, target_shape, target_color, _) = self.process_targets(targets[k][l], self.dataset)

                    if pred_object_size == target_object_size:
                        size_count += 1
                    if pred_shape == target_shape:
                        shape_count += 1
                    if pred_color == target_color:
                        color_count += 1
                    

        return color_count/num_objs, shape_count/num_objs, size_count/num_objs


    def get_alignement_indices(self, preds, targets):
        """
            Returns the optimal indices for the aligment between the predicted slots
            and thez real objects.
            The relation is :
                preds[sigma[i]] <-> targets[i]
        """

        loss = torch.nn.SmoothL1Loss(reduction = 'none')
        pairwise_cost = torch.sum(loss(torch.unsqueeze(preds, axis = -3), torch.unsqueeze(targets, axis = -2)), axis = -1)
        pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
        indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))
        #indices shape [batch_size, 2, num_slots]

        return indices[:,1,:]

