# License: MIT
# Author: Karl Stelzner

from collections import namedtuple
import os

config_options = [
    # Training config
    'vis_every',  # Visualize progress every X iterations
    'batch_size',
    'num_epochs',
    'load_parameters',  # Load parameters from checkpoint
    'checkpoint_file',  # File for loading/storing checkpoints
    'data_dir',  # Directory for the training data
    'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'dim_points', # Dim of the caracteritics of the objects we want to predict
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
]

MonetConfig = namedtuple('MonetConfig', config_options)

sprite_config = MonetConfig(vis_every=50,
                            batch_size=8,
                            num_epochs=20,
                            load_parameters=True,
                            checkpoint_file='./checkpoints/sprites.ckpt',
                            data_dir='./data/',
                            parallel=False,
                            num_slots=4,
                            num_blocks=5,
                            channel_base=64,
                            dim_points = 3,
                            bg_sigma=0.09,
                            fg_sigma=0.11,
                           )
                           
multi_sprite_config = MonetConfig(vis_every=50,
                            batch_size=8,
                            num_epochs=30,
                            load_parameters=True,
                            checkpoint_file='./checkpoints/multi_sprites_supervised.ckpt',
                            data_dir='../../Datasets/MultiObjectDatasetCreator/generated/dsprites/multi_dsprites_210407_144010.npz',
                            parallel=False,
                            num_slots=4,
                            num_blocks=5,
                            channel_base=64,
                            dim_points = 3,
                            bg_sigma=0.09,
                            fg_sigma=0.11,
                           )
                           
multi_sprite_rela_config = MonetConfig(vis_every=50,
                            batch_size=8,
                            num_epochs=30,
                            load_parameters=True,
                            checkpoint_file='./checkpoints/multi_sprites_supervised_relation_1.ckpt',
                            data_dir='./data/multi_dsprites_top_right_top_and_right_monet.npz',
                            parallel=False,
                            num_slots=4,
                            num_blocks=5,
                            channel_base=64,
                            dim_points = 3,
                            bg_sigma=0.09,
                            fg_sigma=0.11,
                           )

clevr_config = MonetConfig(vis_every=1000,
                           batch_size=4,
                           num_epochs=100,
                           load_parameters=True,
                           checkpoint_file='./checkpoints/clevr64.ckpt',
                           data_dir=os.path.expanduser('./data/CLEVR_v1.0/images/train/'),
                           parallel=True,
                           num_slots=11,
                           num_blocks=6,
                           channel_base=64,
                           dim_points = 3,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                          )



