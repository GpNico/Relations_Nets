# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import scipy.optimize


class TrainingMonitor:

    def __init__(self, dataset = 'clevr'):
        
        self.dataset = dataset
        self.sigmas = None


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
        
        self.sigmas = sigmas

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
        
    def get_rela_precision(self, outputs, targets):
        """
        Args:
            outputs : dict (outputs_rela) [batch_size, num_slots*(num_slots - 1), dim_rela]
            targets : tensor [batch_size, num_slots, num_slots, dim_rela]
        Returns:
            precision : float
        """
        
        preds = outputs['outputs_rela'].detach().cpu().numpy()
        index_to_pair = outputs['index_to_pair']
        targets = targets.detach().cpu().numpy()
        
        rela_count = 0
        
        num_rela = 0
        
        sigmas = self.sigmas
        
        batch_size, num_slots, _, dim_rela = targets.shape
        
        #Computing sigma_inv
        sigmas_inv = np.zeros_like(sigmas)
        batch_size, num_slots = sigmas.shape
        for k in range(batch_size):
            for l in range(num_slots):
                sigmas_inv[k, sigmas[k, l]] = l
        
        #fundamental relation : outputs_rela[n, i] ~ rela_labels[n, sigma_inv[index_to_pair[i][0]], sigma_inv[index_to_pair[i][1]]]
        for n in range(batch_size):
            for i in range(num_slots*(num_slots-1)):
                pred = preds[n, i]
                target = targets[n, sigmas_inv[n,index_to_pair[i][0]], sigmas_inv[n,index_to_pair[i][1]]]
                
                if target.max() > 0: #there is a relation
                    num_rela += 1
                    
                    pred_idx = np.argmax(pred)
                    target_idx = np.argmax(target)
                    if pred_idx == target_idx:
                        rela_count += 1
                                       
        return rela_count/num_rela


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

