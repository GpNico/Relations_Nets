# License: MIT
# Author: Karl Stelzner

import os 
import argparse

import tqdm

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='supervised.yml',
                        dest='config',
                        help='config you wish to load')
                        
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='multi_sprite',
                        dest='dataset',
                        help='which dataset to load : sprite, multi_sprite, clevr, multi_sprite_equal')
                        
    parser.add_argument('-p',
                        '--prediction',
                        type=str,
                        default='all_carac',
                        dest='prediction',
                        help='which dataset to make (only in supervised training)')
    
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='slot_att',
                        dest='model',
                        help='which model to load : monet or slot_att')

    parser.add_argument('-n',
                        '--num_run',
                        type=int,
                        default=5,
                        dest='num_run',
                        help='number of run to execute !')
    
    args = parser.parse_args()

    for run in tqdm.tqdm(range(args.num_run)):
        os.system(f'python main.py -c {args.config} -d {args.dataset} -p {args.prediction} -m {args.model} -s')
        

    

