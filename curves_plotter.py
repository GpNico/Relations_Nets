# License: MIT
# Author: Karl Stelzner


import numpy as np 
import os
import matplotlib.pyplot as plt 
import pickle


plt.rcParams.update({'font.size': 22})

filename = 'contact_experiment'
savefile = open(filename, 'rb')

list_of_dict = pickle.load(savefile)

savefile.close()   

rela_names = ['right', 'left', 'top', 'below', 'contact right', 'contact left', 'contact top', 'contact below']

rela_precision, rela_recall, rela_f1 = [], [], []
for k in range(len(list_of_dict)):
    rela_precision.append(list_of_dict[k]['rela_precision'])
    rela_recall.append(list_of_dict[k]['rela_recall'])
    rela_f1.append(list_of_dict[k]['rela_f1'])
rela_precision = np.array(rela_precision)
rela_recall = np.array(rela_recall)
rela_f1 = np.array(rela_f1)

X = np.arange(rela_precision.shape[1])

plt.figure(figsize=(12,12))
plt.title('Relations Precision over Learning')
for k in range(rela_precision.shape[2]):
    plt.plot(X, rela_precision[:,:,k].mean(axis = 0), label=rela_names[k])
    plt.fill_between(X, rela_precision[:,:,k].mean(axis = 0) - rela_precision[:,:,k].std(axis = 0), rela_precision[:,:,k].mean(axis = 0) + rela_precision[:,:,k].std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('Precision')
#plt.show()

X = np.arange(rela_recall.shape[1])

plt.figure(figsize=(12,12))
plt.title('Relations Recall over Learning')
for k in range(rela_recall.shape[2]):
    plt.plot(X, rela_recall[:,:,k].mean(axis = 0), label=rela_names[k])
    plt.fill_between(X, rela_recall[:,:,k].mean(axis = 0) - rela_recall[:,:,k].std(axis = 0), rela_recall[:,:,k].mean(axis = 0) + rela_recall[:,:,k].std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('Recall')
#plt.show()

X = np.arange(rela_f1.shape[1])

plt.figure(figsize=(12,12))
plt.title('Relations F1 Score over Learning')
for k in range(rela_f1.shape[2]):
    plt.plot(X, rela_f1[:,:,k].mean(axis = 0), label=rela_names[k])
    plt.fill_between(X, rela_f1[:,:,k].mean(axis = 0) - rela_f1[:,:,k].std(axis = 0), rela_f1[:,:,k].mean(axis = 0) + rela_f1[:,:,k].std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('F1 Score')
plt.show()

X = np.arange(rela_precision.shape[1])

rela_type = ['no contact', 'contact']

plt.figure(figsize=(12,12))
plt.title('Relations Precision over Learning')
for k in range(2):
    plt.plot(X, rela_precision[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0), label=rela_type[k])
    plt.fill_between(X, rela_precision[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) - rela_precision[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), rela_precision[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) + rela_precision[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('Precision')
#plt.show()

X = np.arange(rela_recall.shape[1])

plt.figure(figsize=(12,12))
plt.title('Relations Recall over Learning')
for k in range(2):
    plt.plot(X, rela_recall[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0), label=rela_type[k])
    plt.fill_between(X, rela_recall[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) - rela_recall[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), rela_recall[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) + rela_recall[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('Recall')
#plt.show()

X = np.arange(rela_f1.shape[1])

plt.figure(figsize=(12,12))
plt.title('Relations F1 Score over Learning')
for k in range(2):
    plt.plot(X, rela_f1[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0), label=rela_type[k])
    plt.fill_between(X, rela_f1[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) - rela_f1[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), rela_f1[:,:,4*k:4*(k+1)].mean(axis = 2).mean(axis = 0) + rela_f1[:,:,4*k:4*(k+1)].mean(axis = 2).std(axis = 0), alpha = 0.2)
plt.legend(loc='best')
plt.xlabel('Training Steps (x250)')
plt.ylabel('F1 Score')
plt.show()