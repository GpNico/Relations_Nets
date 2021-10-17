# License: MIT
# Author: Karl Stelzner


import visdom
import os
import argparse

import src.train as train 
import src.utils as utils

import warnings


with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        vis = utils.VisdomPlotter(env_name = 'Plot Monitor')
        print("Visdom session detected !")
    except Warning: 
        print("No visdom session detected !")
        vis = False
        

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

    parser.add_argument('-w',
                        '--warmup',
                        type=int,
                        default=0,
                        dest='warmup',
                        help='Do we have a phase were we focus on learning object characteristics or do we start to learn both at the same time. If yes warmup contains the number of epoch we keep alpha = 0.')

    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=50,
                        dest='epochs',
                        help='Number of epochs.')
    
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='slot_att',
                        dest='model',
                        help='which model to load : monet or slot_att')

    parser.add_argument("-s",
                        "--save_data", 
                        help="whether or not we save the training data",
                        action="store_true")

    parser.add_argument("-f",
                        "--file_name",
                        type=str,
                        default='',
                        dest='file_name',
                        help="name of the net weights save.")

    
    args = parser.parse_args()
    
    args.config = os.path.join('config', args.config)
    
    if 'unsupervised' in args.config: 
        train.unsupervised_experiment(args, vis)
    else:
        train.supervised_experiment(args, vis)
    

