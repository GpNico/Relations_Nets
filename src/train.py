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

#################################
#                               #
#          EXPERIMENTS          #
#                               #
#################################


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
    trainset = ObjectDataset(config['prediction'][args.prediction]['filepath'], train=True, transform= transform)
    
    valset = ObjectDataset(config['prediction'][args.prediction]['filepath'], train=False, transform= transform)
    
    trainloader = DataLoader(trainset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=0)
    
    valoader = DataLoader(trainset, batch_size=config['params']['batch_size'], shuffle=False, num_workers=0)
    
    #Create Model
    try:
        dim_rela = config['prediction'][args.prediction]['dim_rela']
        print("Predicting Relations ... ")
    except:
        print("Not Predicting Relations ...")
        dim_rela = 0
    
    if args.model == 'monet':
        model_net = model.MonetClassifier(config['params'], config['params']['height'],
                                                        config['params']['width'],
                                                        config['prediction'][args.prediction]['dim_points'],
                                                        dim_rela).cuda()
    elif args.model == 'slot_att':
        model_net = model.SlotAttentionClassifier(config['params'], config['params']['height'],
                                                        config['params']['width'],
                                                        config['prediction'][args.prediction]['dim_points'],
                                                        dim_rela).cuda()
    else:
        raise Exception("Error in the model name. Make sure it is in {monet, slot_att}.")
    
    print("Start Training")
    #Run Training
    run_training_supervised(model_net, config, args.dataset, args.prediction, trainloader, valoader, vis)
    
    
    
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
                
                vis.plotline('loss', 'train', 'Loss', epoch*iter_per_epoch + i, running_loss / conf['vis_every'] )
                vis.plotimagemask('recons', utils.numpify(images[:8]), 
                                            utils.numpify(output['masks'][:8]),
                                            utils.numpify(output['reconstructions'][:8]) )
                
                
                running_loss = 0.0
                
        torch.save(monet.state_dict(), conf['checkpoint_file'])

    print('training done')
    
    
def run_training_supervised(model, conf, dataset, pred, trainloader, valoader, vis):
    if conf['params']['load_parameters'] and os.path.isfile(conf['prediction'][pred]['checkpoint_file']):
        model.load_state_dict(torch.load(conf['prediction'][pred]['checkpoint_file']))
        print('Restored parameters from', conf['prediction'][pred]['checkpoint_file'])
    else:
        for w in model.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')
        
    output_to_title = utils.import_from_path(conf['prediction'][pred]['output_to_title']['filepath'],
                                             conf['prediction'][pred]['output_to_title']['fct'])
                                             
    get_ground_truth = utils.import_from_path(conf['prediction'][pred]['get_ground_truth']['filepath'],
                                              conf['prediction'][pred]['get_ground_truth']['fct'])

    #optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    criterion = utils.hungarian_huber_loss
    
    iter_per_epoch = len(trainloader)

    for epoch in range(conf['params']['num_epochs']):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.cuda(), get_ground_truth(labels, conf, pred).cuda()

            optimizer.zero_grad()
            output, loss = model.get_loss(images, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % conf['params']['vis_every'] == conf['params']['vis_every']-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / conf['params']['vis_every']))
                
                vis.plotline('loss', 'train', 'Loss', epoch*iter_per_epoch + i, running_loss / conf['params']['vis_every'] )

                vis.plotimage('image1', utils.numpify(images[0]), output_to_title(output[0]))
                vis.plotimage('image2', utils.numpify(images[1]), output_to_title(output[1]))
                vis.plotimage('image3', utils.numpify(images[2]), output_to_title(output[2]))
                vis.plotimage('image4', utils.numpify(images[3]), output_to_title(output[3]))

                model.eval()

                ap_train = [utils.average_precision(output.detach().cpu().numpy(), labels.detach().cpu().numpy(), d, dataset) for d in [-1., 1., 0.5, 0.25, 0.125] ]

                images, labels = iter(valoader).next()
                labels = get_ground_truth(labels, conf, pred)
                output, _ = model(images.cuda())
                ap_val = [utils.average_precision(output.detach().cpu().numpy(), labels.detach().numpy(), d, dataset) for d in [-1., 1., 0.5, 0.25, 0.125] ]
                
                vis.plotline('AP', 'train', 'Average Precision', epoch*iter_per_epoch + i, ap_train[1] )
                vis.plotline('AP', 'val', 'Average Precision', epoch*iter_per_epoch + i, ap_val[1] )
                
                running_loss = 0.0
                model.train()
                
        torch.save(model.state_dict(), conf['prediction'][pred]['checkpoint_file'])

    print('training done')

