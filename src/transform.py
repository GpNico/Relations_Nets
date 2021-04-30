# License: MIT
# Author: Karl Stelzner

import torchvision.transforms as transforms


def identity(x):
    return x

def to_tensor(x):
    return transforms.ToTensor()(x)
    
def functional_crop(x):
    return transforms.functional.crop(x, 29, 64, 192, 192)

def get_first_three(x):
    return x[:3]
    
def compose_1(x):
    crop_tf = transforms.Lambda(functional_crop)
    drop_alpha_tf = transforms.Lambda(get_first_three)
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf
                                   ])
    return transform(x)

