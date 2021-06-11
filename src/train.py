# License: MIT
# Author: Karl Stelzner

import torch
import torch
import torch.nn as nn
import torch.optim as optim

import os
import yaml

import src.model as model
import src.utils as utils
from src.training_monitor import TrainingMonitor

#################################
#                               #
#          EXPERIMENTS          #
#                               #
#################################

#To reproduce results
#torch.manual_seed(7)

def unsupervised_experiment(args, vis):
    
    #Get config file
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)['datasets'][args.dataset]
    
    # Get Objects and Functions
    ObjectDataset = utils.import_from_path(config['objectdataset']['filepath'], config['objectdataset']['class'])
        
    transform = utils.import_from_path(config['transform']['filepath'], config['transform']['fct'])
    
    if config['dataloader']['class'] == 'default':
        DataLoader = torch.utils.data.DataLoader
    else:
        DataLoader = utils.import_from_path(config['dataloader']['filepath'], config['dataloader']['class'])

    #Create DataLoader
    trainset = ObjectDataset(config['data']['filepath'], train=True, transform= transform)
    
    trainloader = DataLoader(trainset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)
    
    #Create Model
    monet = model.Monet(config['params'], config['params']['height'], config['params']['width']).cuda()
    
    print("Start Training")
    #Run Training
    run_training_unsupervised(monet, config['params'], trainloader, vis)
    
    
def supervised_experiment(args, vis):
    print("Reading Config File ...")
    #Get config file
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)['datasets'][args.dataset]
    
    # Get Objects and Functions
    ObjectDataset = utils.import_from_path(config['objectdataset']['filepath'], config['objectdataset']['class'])
        
    transform = utils.import_from_path(config['transform']['filepath'], config['transform']['fct'])
    
    if config['dataloader']['class'] == 'default':
        DataLoader = torch.utils.data.DataLoader
    else:
        DataLoader = utils.import_from_path(config['dataloader']['filepath'], config['dataloader']['class'])
    
    print("Loading Data ...")
    #Create DataLoader
    trainset = ObjectDataset(config['prediction'][args.prediction]['filepath'], train=True, transform= transform)
    
    valset = ObjectDataset(config['prediction'][args.prediction]['filepath'], train=False, transform= transform)
    
    trainloader = DataLoader(trainset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)
    
    valoader = DataLoader(valset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)
    
    #Create Model
    if 'rela' in args.prediction:
        dim_rela = config['prediction'][args.prediction]['dim_rela']
        print("Predicting Relations ... ")
        
        model_net = model.RelationsPredictor(config['params'],
                                         config['params']['height'],
                                         config['params']['width'],
                                         config['prediction'][args.prediction]['dim_points'], 
                                         config['prediction'][args.prediction]['dim_rela'], 
                                         object_classifier = args.model,
                                         load_params = config['prediction'][args.prediction]).cuda()
        
    else:
        print("Not Predicting Relations ...")

        if args.model == 'monet':
            model_net = model.MonetClassifier(config['params'], config['params']['height'],
                                              config['params']['width'],
                                              config['prediction'][args.prediction]['dim_points']).cuda()
        elif args.model == 'slot_att':
            model_net = model.SlotAttentionClassifier(config['params'], config['params']['height'],
                                                      config['params']['width'],
                                                      config['prediction'][args.prediction]['dim_points']).cuda()
        else:
            raise Exception("Error in the model name. Make sure it is in {monet, slot_att}.")
    
    print("Start Training")
    #Run Training
    run_training_supervised(model_net, config, args.dataset, args.prediction, trainloader, valoader, vis, args.save_data)
    
    
    
#################################
#                               #
#        TRAINING RUNS          #
#                               #
#################################
    
    
def run_training_unsupervised(monet, conf, trainloader, vis):
    if conf['load_parameters'] and os.path.isfile(conf['checkpoint_file']):
        monet.load_state_dict(torch.load(conf['checkpoint_file']))
        print('Restored parameters from', conf['checkpoint_file'])
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)
    
    iter_per_epoch = len(trainloader)

    for epoch in range(conf['num_epochs']):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % conf['vis_every'] == conf['vis_every']-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / conf['vis_every']))
                
                try:
                    vis.plotline('loss', 'train', 'Loss', epoch*iter_per_epoch + i, running_loss / conf['vis_every'] )
                    vis.plotimagemask('recons', utils.numpify(images[:8]), 
                                                utils.numpify(output['masks'][:8]),
                                                utils.numpify(output['reconstructions'][:8]) )
                except:
                    pass
                
                running_loss = 0.0
                
        torch.save(monet.state_dict(), conf['checkpoint_file'])

    print('training done')
    
    
def run_training_supervised(model, conf, dataset, pred, trainloader, valoader, vis, save_data):
    if conf['params']['load_parameters'] and os.path.isfile(conf['prediction'][pred]['checkpoint_file']):
        model.load_state_dict(torch.load(conf['prediction'][pred]['checkpoint_file']))
        print('Restored parameters from', conf['prediction'][pred]['checkpoint_file'])
    else:
        for name, w in model.named_parameters():
            continue
        print('Initialized parameters')
                                             
    get_ground_truth = utils.import_from_path(conf['prediction'][pred]['get_ground_truth']['filepath'],
                                              conf['prediction'][pred]['get_ground_truth']['fct'])


    training_monitor = TrainingMonitor(pred, dataset)


    base_learning_rate = conf['params']['learning_rate']
    warmup_steps, decay_rate, decay_steps = conf['params']['warmup_steps'], conf['params']['decay_rate'], conf['params']['decay_steps']
    global_step = 0

    def learning_rate_scheduler(step):
        if step < warmup_steps:
            factor = step/warmup_steps
        else:
            factor = 1 
        
        factor = factor * ( decay_rate**(step/decay_steps) )
 
        return factor

    optimizer = optim.Adam(model.parameters(), lr = base_learning_rate,  eps=1e-08)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scheduler)
    criterion = utils.hungarian_huber_loss

    for epoch in range(conf['params']['num_epochs']):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.cuda(), get_ground_truth(labels, conf, pred)

            optimizer.zero_grad()
            dict, loss = model.get_loss(images, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            scheduler.step()

            with torch.no_grad():
                utils.training_loop_validation(model, conf, global_step, epoch, running_loss, vis, valoader, get_ground_truth, pred, training_monitor, dataset, optimizer)

            
            global_step += 1
                
        torch.save(model.state_dict(), conf['prediction'][pred]['checkpoint_file'])


    if save_data:
        training_monitor.save_to_pickle()

    print('training done')

