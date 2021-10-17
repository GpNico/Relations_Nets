# License: MIT
# Author: Karl Stelzner

import os 
import argparse
import yaml
from importlib import import_module

import tqdm

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


def label_analyser(N_label, dataloader):
    label_counter = [0 for k in range(N_label)]

    for batch in tqdm.tqdm(dataloader, total = len(dataloader)):
      _, labels = batch
      label = labels['relation'].view(-1, 8)
      for k in range(label.shape[0]):
        target = label[k].cpu().detach().tolist()
        if max(target) > 0:
          for l in range(N_label):
              if target[l] == 1:
                  label_counter[l] += 1
    for l in range(N_label):
        print("label {} count : {}".format(l, label_counter[l]))

def get_dataloaders(args):
    print("Reading Config File ...")
    path_to_config = '..\\' + args.config
    #Get config file
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)['datasets'][args.type]
    # Get Objects and Functions
    ObjectDataset = import_from_path(config['objectdataset']['filepath'][4:], config['objectdataset']['class'])
        
    transform = import_from_path(config['transform']['filepath'][4:], config['transform']['fct'])

    if config['dataloader']['class'] == 'default':
        DataLoader = torch.utils.data.DataLoader
    else:
        DataLoader = import_from_path(config['dataloader']['filepath'][4:], config['dataloader']['class'])
    print("Loading Data ...")
    #Create DataLoader
    trainset = ObjectDataset(config['prediction'][args.prediction][args.dataset]['filepath'], train=True, transform= transform)
    
    valset = ObjectDataset(config['prediction'][args.prediction][args.dataset]['filepath'], train=False, transform= transform)
    
    trainloader = DataLoader(trainset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)
    
    valoader = DataLoader(valset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)

    dim_rela = config['prediction'][args.prediction][args.dataset]['dim_rela']

    return trainloader, valoader, dim_rela

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='supervised.yml',
                        dest='config',
                        help='config you wish to load')
                        
    parser.add_argument('-t',
                        '--type',
                        type=str,
                        default='multi_sprite',
                        dest='type',
                        help='which type dataset to use : multi_sprite, clevr')

    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='rela_contact',
                        dest='dataset',
                        help='which precise dataset to load : all_carac, rela_contact, R_L_CT_CB, T_B_CR_CL')
                        
    parser.add_argument('-p',
                        '--prediction',
                        type=str,
                        default='all_carac',
                        dest='prediction',
                        help='which dataset to make (only in supervised training)')
    
    args = parser.parse_args()

    args.config = os.path.join('config', args.config)

    trainloader, valoader, N_label = get_dataloaders(args)

    print("## TRAINSET ##")
    label_analyser(N_label, trainloader)
    print('')
    print("## VALSET ##")
    label_analyser(N_label, valoader)
    print('')
        

    

