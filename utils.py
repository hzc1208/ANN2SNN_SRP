import torch
import torch.nn as nn
from modules import *


def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False

    
def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = IFNeuron(scale=module.v_threshold)
    return model


def replace_activation_by_MPLayer(model, presim_len, sim_len):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_MPLayer(module, presim_len, sim_len)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = MPLayer(v_threshold=module.up.item(), presim_len=presim_len, sim_len=sim_len)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=8., t=t)
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def error(info):
    print(info)
    exit(1)
    
