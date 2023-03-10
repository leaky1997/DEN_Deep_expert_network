from inspect import signature
import torch
from torch import nn

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

def get_arity(f):
    return len(signature(f).parameters) # 参数数量



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1)
            m.bias.data.zero_()

            m.bias.data.zero_()
def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.rand(x.size()).cuda() * torch.sqrt(npower)
def strike(x, snr):
    "加冲击噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    
    pass

    
    return torch.rand(x.size()).cuda() * torch.sqrt(npower)



class bearing_signal_simulation():
    """
        original matlab parameter
        fs = 3e3
        t = np.arange(1/fs, 2, 1/fs)# /fs
        fr = 600;                       #% Carrier signal
        fd = 13;                        #% discrete signal
        ff = 10;                        #% Characteristic frequency(Modulating signal)
        a = 0.02;                       #% Damping ratio
        T = 1/ff;                       #% Cyclic period
        fs = 3e3;                       #% Sampling rate
        K = 1;                         #% Number of impulse signal
        K = np.linspace(0,50,5999)
        t = [1/fs:1/fs:2]                #% Time
        A=5;                            #% Maximum amplitude    
        noise = 0.5
    """
    def __init__(
        self,fs = 3e3,time = 2,fr = 600,fd = 13,
        ff = 10,damping_ratio = 0.02,K = 50,A = 5,noise = 0.5
    ):
        self.t = np.arange(1/fs, time, 1/fs)
        self.K = np.linspace(0,K,len(self.t))
        self.t_impact = np.mod(self.t,1/ff)
        self.fs = fs
        self.A = A
        self.fr = fr
        self.fd = fd
        self.ff = ff
        self.damping_ratio = damping_ratio
        self.noise = noise
        self.signal_generatator()
    def signal_generatator(self):        
        self.x1 = self.A*np.exp(-self.damping_ratio*2*math.pi*self.fr*(self.t_impact))
        self.x2 = np.sin(2*math.pi*self.fr*(self.t))
        self.x3 = self.x1*self.x2
        self.x4 = np.random.normal(0,self.noise,len(self.x3))
        self.x5 = 2*np.sin(2*math.pi*self.t*self.fd)
        self.vib = self.x3+self.x4+self.x5
        return self.vib
    def signal_plot(self,signal = 'vib'):
        plt.figure()
        plt.plot(self.vib,label = 'vib')
if __name__ == '__main__':   
    bearing1 = bearing_signal_simulation(fs = 3e3,time = 2,fr = 600,
        fd = 13,ff = 20,damping_ratio = 0.02,K = 50,A = 0,noise = 2)
    bearing2 = bearing_signal_simulation(fs = 3e3,time = 2,fr = 600,
        fd = 13,ff = 40,damping_ratio = 2,K = 50,A = 1,noise = 2)
    bearing3 = bearing_signal_simulation(fs = 3e3,time = 2,fr = 600,
        fd = 13,ff = 80,damping_ratio = 2,K = 50,A = 2,noise = 2)
    bearing4 = bearing_signal_simulation(fs = 3e3,time = 2,fr = 600,
        fd = 13,ff = 160,damping_ratio = 2,K = 50,A = 3,noise = 2)
    bearing1.signal_plot()
    bearing2.signal_plot()
    bearing3.signal_plot()
    bearing4.signal_plot()
        
