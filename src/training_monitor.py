# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import scipy.optimize

import pickle
import os

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class TrainingMonitor:

    def __init__(self, pred, dataset = 'clevr'):
        
        self.dataset = dataset
        self.pred = pred
        self.sigmas = None

        self.rela_precision_list = []
        self.rela_recall_list = []
        self.rela_f1_list = []

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
        elif data == 'multi_sprite_equal':
            shape = np.argmax(target[:5])
            object_size = np.argmax(target[5:10])
            material = 1. #we do not predict material with multi_sprite
            color = np.argmax(target[10:15])
            coords = target[15:17]
            real_obj = target[17]
            
        return coords, object_size, material, shape, color, real_obj

    def get_carac_precision(self, preds, targets):
        """
        Args:
            preds : tensor [batch_size, num_slots, dim_points]
            targets : tensor [batch_size, num_slots, dim_points]
        Returns:
            precision : float
        """

        color_true, color_pred = [], []
        shape_true, shape_pred = [], []
        size_true, size_pred = [], []

        sigmas = self.get_alignement_indices(preds, targets)
        
        self.sigmas = sigmas

        preds, targets = preds.detach().cpu().numpy(), targets.detach().cpu().numpy()
        
        batch_size, num_slots = sigmas.shape

        for k in range(batch_size):
            for l in range(num_slots):

                if targets[k][l].max() > 0.: #There is an object in this slot

                    (_, pred_object_size, pred_material, pred_shape, pred_color, _) = self.process_targets(preds[k][sigmas[k][l]], self.dataset)
                    (_, target_object_size, target_material, target_shape, target_color, _) = self.process_targets(targets[k][l], self.dataset)

                    #if (pred_object_size, pred_material, pred_shape, pred_color) == (target_object_size, target_material, target_shape, target_color):

                    color_true.append(target_color)
                    color_pred.append(pred_color)
                    shape_true.append(target_shape)
                    shape_pred.append(pred_shape)
                    size_true.append(target_object_size)
                    size_pred.append(pred_object_size)

        
        cm_color = confusion_matrix(color_true, color_pred)
        cm_shape = confusion_matrix(shape_true, shape_pred)
        cm_size = confusion_matrix(size_true, size_pred)

        color_precision, color_recall, color_f1, _ = precision_recall_fscore_support(color_true, color_pred, average='macro')
        shape_precision, shape_recall, shape_f1, _ = precision_recall_fscore_support(shape_true, shape_pred, average='macro')
        size_precision, size_recall, size_f1, _ = precision_recall_fscore_support(size_true, size_pred, average='macro')

        metric = {'confusion_matrix': (cm_color, cm_shape, cm_size),
                  'carac_precision': (color_precision, shape_precision, size_precision),
                  'carac_recall': (color_recall, shape_recall, size_recall),
                  'carac_f1': (color_f1, shape_f1, size_f1)}

        return metric
        
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
        
        sigmas = self.sigmas
        
        batch_size, num_slots, _, dim_rela = targets.shape

        ####### COMPUTING METRIC #######
        rela_tp = np.zeros(dim_rela)
        rela_fp = np.zeros(dim_rela)
        rela_fn = np.zeros(dim_rela)
        rela_tn = np.zeros(dim_rela)
        
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
                
                    pred_idx = np.where(pred > 0.5)[0]
                    target_idx = np.where(target > 0.5)[0]

                    for k in range(dim_rela):
                        if k in pred_idx:
                            if k in target_idx:
                                rela_tp[k] += 1
                            else:
                                rela_fp[k] += 1
                        else:
                            if k in target_idx:
                                rela_fn[k] += 1
                            else:
                                rela_tn[k] += 1
                                
        #Precision
        rela_precision = np.zeros(dim_rela)
        rela_recall = np.zeros(dim_rela)
        rela_f1 = np.zeros(dim_rela)
        for k in range(dim_rela):
            if rela_tp[k] != 0:
                rela_precision[k] = rela_tp[k]/(rela_tp[k] + rela_fp[k])
                rela_recall[k] = rela_tp[k]/(rela_tp[k] + rela_fn[k])
                rela_f1[k] = rela_tp[k]/(rela_tp[k] + 0.5*(rela_fp[k] + rela_fn[k]))
            else:
                rela_precision[k] = 0
                rela_recall[k] = 0
                rela_f1[k] = 0

        metrics = {'rela_tp': rela_tp,
                   'rela_fp': rela_fp,
                   'rela_fn': rela_fn,
                   'rela_tn': rela_tn,
                   'rela_precision': rela_precision,
                   'rela_recall': rela_recall,
                   'rela_f1': rela_f1}

        self.rela_precision_list.append(rela_precision)
        self.rela_recall_list.append(rela_recall)
        self.rela_f1_list.append(rela_f1)
                                       
        return metrics


    def get_alignement_indices(self, preds, targets):
        """
            Returns the optimal indices for the aligment between the predicted slots
            and thez real objects.
            The relation is :
                preds[sigma[i]] <-> targets[i]
        """
        num_slot = preds.shape[1]
        loss = torch.nn.SmoothL1Loss(reduction = 'none')
        pairwise_cost = torch.sum(loss(torch.unsqueeze(preds, axis = -3).expand(-1, num_slot, -1,-1), 
                                       torch.unsqueeze(targets, axis = -2).expand(-1, -1, num_slot, -1)), axis = -1)
        pairwise_cost_np = pairwise_cost.cpu().detach().numpy()
    
        indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost_np)))
        #indices shape [batch_size, 2, num_slots]

        return indices[:,1,:]

    def save_to_pickle(self):
        
        #Data

        dict_data = {'rela_precision': self.rela_precision_list,
                     'rela_recall': self.rela_recall_list,
                     'rela_f1': self.rela_f1_list}

        #Dump
        filename = 'contact_experiment_1'
                
        if not(os.path.exists(filename)):
            print("Creating Save File ...")
            savefile = open(filename, 'wb')
            pickle.dump([], savefile)
            savefile.close()
        
        savefile = open(filename, 'rb')

        list_of_dict = pickle.load(savefile)

        savefile.close()

        list_of_dict.append(dict_data)

        savefile = open(filename, 'wb')
        pickle.dump(list_of_dict, savefile)
        savefile.close()

    def is_equal_arr(self, arr1, arr2):
        l1 = arr1.shape[0]
        l2 = arr2.shape[0]
        if not(l1 == l2):
            return False
        for k in range(l1):
            if not(arr1[k] == arr2[k]):
                return False
        return True



