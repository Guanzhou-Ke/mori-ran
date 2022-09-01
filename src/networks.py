"""
Network building tools.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import torch
from torch import nn


def build_mlp(layers, activation='relu', norm='batch'):
    """Build multiple linear perceptron

    Args:
        layers (list): The list of input and output dimension.
        activation (str, optional): activation function. Defaults to 'relu'.
                                    ['none', 'relu', 'softmax', 'sigmoid']
        norm (str, optional): normalization. Defaults to 'batch'.
                              `none`: not set, `batch`: denotes BatchNorm1D;
                              `layer`: denotes LayerNorm.
    """
    net = []
    for idx in range(1, len(layers)):
        net.append(
            nn.Sequential(
                nn.Linear(layers[idx-1], layers[idx]),
                _get_norm(norm, num_features=layers[idx], dim=1),
                _get_act(activation),
            )
        )
    net = nn.Sequential(*net)
    return net
        
def _get_act(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(dim=-1)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Activation function name error: {name}")
    
def _get_norm(name, num_features, dim=1):
    if name == None:
        return nn.Identity()
    elif name == 'batch':
        return nn.BatchNorm1d(num_features) if dim == 1 else nn.BatchNorm2d(num_features)
    elif name == 'layer':
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Normalization name erro: {name}")
    

if __name__ == '__main__':
    layers = [1024, 225, 512, 128]
    act = 'softmax'
    norm = 'batch'
    net = build_mlp(layers, activation=act, norm=norm)
    print(net)
    x = torch.rand(10, 1024)
    y = net(x)
    print(y.shape)