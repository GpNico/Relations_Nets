# Relations_Nets

This repo has been created in the context of an internship at LSCP on Attention Mechanism. The goal is to use the MONet architechture to study its capacities of learning relations between objects. The overall idea is to investigate whether or not a deep attention mechanism can learn more easily natural relationship between objects, in the sense that the word that describes this relation exists in the human language.

# Important

This work is based on the PyTorch implementation of MONet from this repositery :

https://github.com/stelzner/monet

For the Slot Attention implementation it's a mix between :

https://github.com/lucidrains/slot-attention
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
