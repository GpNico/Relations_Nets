# License: MIT
# Author: Karl Stelzner

import numpy as np
import torch
import scipy.optimize

import pickle
import os

from sklearn.metrics import confusion_matrix


class TrainingMonitor:

    def __init__(self, pred, dataset = 'clevr'):
        
        self.dataset = dataset
        self.pred = pred
        self.sigmas = None

        self.color_prec = []
        self.shape_prec = []
        self.size_prec = []
        self.overall_prec = []
        self.rela_prec = []
        self.rela_contact_prec = []
        self.rela_no_contact_prec = []


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

        color_count = 0
        shape_count = 0
        size_count = 0

        pred_objs = 0

        color_true, color_pred = [], []
        shape_true, shape_pred = [], []
        size_true, size_pred = [], []

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

                    if (pred_object_size, pred_material, pred_shape, pred_color) == (target_object_size, target_material, target_shape, target_color):
                        pred_objs += 1

                    color_true.append(target_color)
                    color_pred.append(pred_color)
                    shape_true.append(target_shape)
                    shape_pred.append(pred_shape)
                    size_true.append(target_object_size)
                    size_pred.append(pred_object_size)

                    if pred_object_size == target_object_size:
                        size_count += 1
                    if pred_shape == target_shape:
                        shape_count += 1
                    if pred_color == target_color:
                        color_count += 1
        
        cm_color = confusion_matrix(color_true, color_pred)
        cm_shape = confusion_matrix(shape_true, shape_pred)
        cm_size = confusion_matrix(size_true, size_pred)

        self.color_prec.append(color_count/num_objs)
        self.shape_prec.append(shape_count/num_objs)
        self.size_prec.append(size_count/num_objs)
        self.overall_prec.append(pred_objs/num_objs)


        metric = {'precision': (color_count/num_objs, shape_count/num_objs, size_count/num_objs),
                  'confusion_matrix': (cm_color, cm_shape, cm_size),
                  'overall_precision': pred_objs/num_objs}

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
        
        rela_count = 0

        if 'contact' in self.pred:
            rela_contact_count, rela_no_contact_count = 0, 0
            num_rela_contact, num_rela_no_contact = 0, 0
        else:
            rela_contact_count, rela_no_contact_count = 0, 0
            num_rela_contact, num_rela_no_contact = 1, 1

        rela_true, rela_pred = [], []
        
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

                    if 'contact' in self.pred:
                        if target_idx in [0,1,2,3]:
                            num_rela_no_contact += 1
                        elif target_idx in [4,5,6,7]:
                            num_rela_contact += 1

                    rela_true.append(target_idx)
                    rela_pred.append(pred_idx)

                    if pred_idx == target_idx:
                        rela_count += 1
                        if 'contact' in self.pred:
                            if target_idx in [0,1,2,3]:
                                rela_no_contact_count += 1
                            elif target_idx in [4,5,6,7]:
                                rela_contact_count += 1

                            

        self.rela_prec.append(rela_count/num_rela)

        if 'contact' in self.pred:
            self.rela_contact_prec.append(rela_contact_count/num_rela_contact)
            self.rela_no_contact_prec.append(rela_no_contact_count/num_rela_no_contact)

        cm_rela= confusion_matrix(rela_true, rela_pred)

        metrics = {'rela_prec': rela_count/num_rela,
                  'confusion_matrix': cm_rela,
                  'rela_contact_prec': rela_contact_count/num_rela_contact,
                  'rela_no_contact_prec': rela_no_contact_count/num_rela_no_contact}
                                       
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

        dict_data = {'color_prec': self.color_prec,
                     'shape_prec': self.shape_prec,
                     'size_prec': self.size_prec,
                     'overall_prec': self.overall_prec,
                     'rela_prec': self.rela_prec,
                     'rela_contact_prec': self.rela_contact_prec,
                     'rela_no_contact_prec': self.rela_no_contact_prec}

        #Dump
        filename = 'debbug_experiment'
                
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



