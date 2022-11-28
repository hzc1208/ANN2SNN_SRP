from torch import nn
import torch.nn.functional as F
import torch
from spikingjelly.clock_driven import neuron
from torch.autograd import Function

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class IFNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(IFNeuron, self).__init__()
        self.v_threshold = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
        
    def forward(self, x):      
        x = x / self.v_threshold               
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5)   
            
        x = self.neuron(x)
        self.t += 1        
        return x * self.v_threshold
        
    def reset(self):
        self.t = 0
        self.neuron.reset()
        

class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

qcfs = FloorLayer.apply

class QCFS(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
    def forward(self, x):
        x = x / self.up
        x = qcfs(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x


class MPLayer(nn.Module):
    def __init__(self, v_threshold, presim_len, sim_len):
        super().__init__()        
        self.neuron = neuron.IFNode(v_reset=None)
        self.v_threshold = v_threshold
        self.t = 0
        self.membrane_lower = None
        self.presim_len = presim_len
        self.sim_len = sim_len
        
 
    def forward(self, x):
          with torch.no_grad():
              if self.t == 0:
                  self.neuron.reset()
                  self.neuron(torch.ones_like(x)*0.5)
              
              output = self.neuron(x/self.v_threshold)
 
              self.t += 1
              
              if self.t == self.presim_len:
                  self.membrane_lower = torch.where(self.neuron.v>1e-3,torch.ones_like(output),torch.zeros_like(output))
                  self.neuron.reset()
                  self.neuron(torch.ones_like(x)*0.5)
              
              if self.t > self.presim_len:
                  output = output * self.membrane_lower
                                                          
              if self.t == self.presim_len + self.sim_len:                     
                  self.t = 0  
                     
              return output*self.v_threshold    

                           