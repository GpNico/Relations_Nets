# Relations_Nets

This repo has been created in the context of an internship at LSCP on Attention Mechanism. The goal is to use the MONet architechture to study its capacities of learning relations between objects. The overall idea is to investigate whether or not a deep attention mechanism can learn more easily natural relationship between objects, in the sense that the word that describes this relation exists in the human language.

# Important

This work is based on the PyTorch implementation of MONet from this repositery :

https://github.com/stelzner/monet

For the Slot Attention implementation it's a mix between :

https://github.com/lucidrains/slot-attention and 
https://github.com/google-research/google-research/tree/master/slot_attention

# Datasets

## Base

### CLEVR

You can download CLEVR dataset with this command line :
```
wget -cN https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
```
Then unzip it and place it in the `data` folder :
```
unzip -q CLEVR_v1.0.zip
```

### Multi-dSprites

To be completed

## Created

To be completed

# How to Use ?

Once you have download the datasets you wanted, you can run an experiment by simply executing :
```
python main.py
```

## Supervised/Unsupervised

To run a unsupervised experiment :
```
python main.py -c unsupervised.yml
```
To run a supervised experiment :
```
python main.py -c supervised.yml
```
*Default : supervised.yml*

## Prediction

The prediction that the network will do can be selected with the -p argument (or --prediction). For example, to predict contact and no-contact relation :
```
python main.py -p rela_contact 
```
*Default : all_carac*

## Type of dataset

To select a multi-dSprite type dataset run :
```
python main.py -t multi_sprite
```
To select a CLEVR type dataset run :
```
python main.py -t clevr
```
*Default: multi_sprite*

## Dataset

The precise dataset can be selected with the -d argument (or --dataset). For example if I want to train the contact vs no-contact experiment I can run :
```
python main.py -p rela_contact -d rela_contact
```
Which seems redundant, but if we want to run it with the R_L_CT_CB dataset it is :
```
python main.py -p rela_contact -d R_L_CT_CB
```
*Default: rela_contact*

## Random Initialisation/Warm-up

The network that predicts relations contains a module that predict the object characteristics and their assignment in slots. This network can be trained alone in the beginning of the training until a certain step specify by the argument -w or --warmup. 

If -w 0 there won't be a warm-up phase in the training and both characteristics and relations will be learned at the same time. If -w > 0 it specifies the network at which step it should start to increase the parameter alpha (parameter that cannot be greater than 0.5). For example, if -w -1 the network will start to increase alpha from the first step.

*Default: 0*

## Model

To select a model use the argument -m (or --model). In practice we only use slot_att. We do not know if monet work except for the unsupervised tasks.
*Default: slot_att*

## Other arguments

Use -s to save data from the training monitor. Do not use in practice unless you are sure of what you are doing. This parameter is used by the multiple_runner.py script.

Use -f (or --file_name) to specify the name of the weights that will be saved. The name will be : prediction + file_name + ".ckpt"

