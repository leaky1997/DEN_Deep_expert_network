"""
Created on Mon OCT 10 19:49:25 2022
@author: Qi Li(李奇)
@Email: liq22@mails.tsinghua.edu.cn
"""

# common modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import sys
sys.path.append("../")

# specific modules
from model.symbolic_base import symbolic_base,convlutional_operator
from sympy import Matrix, Function,simplify
from sympy import hadamard_product, HadamardProduct, MatrixSymbol

# from utils
from utils.file_utils import create_dir
from utils.model_utils import get_arity
from utils.symbolic_utils import prune_matrix,element_mul



def conv1x1(in_channels,out_channels,stride= 1,bias=True,groups=1):
    return nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)

    
class neural_symbolc_base(nn.Module):
    '''
    neural symbolc net 基础
        symbolic_base = None, # 构成网络的符号空间,torch function的 list
        initialization = None,  # 初始化方法，如不给定则采用默认的方法，todo
        input_channel = 1, # 输入维度
        output_channel = None, # 输出维度，一般没用，因为输出维度是根据符号空间确定的
        bias = False, # shift
        device = 'cuda', # 是否使用GPU
        scale = 1, # 符号空间的倍数，例如 输入5维的符号空间，scale = 2 则符号空间为10
        
        该class 可以针对的input 为 [B,C,L], 如果求完均值后的特征可以是[B,C,1]
    '''
    def __init__(self,
                 symbolic_base = None,
                 initialization = None,
                 input_channel = 1,
                 output_channel = None,
                 bias = False,
                 device = 'cuda',
                 scale = 1,
                 kernel_size = None,
                 stride = None,
                 skip_connect = True,
                 amount = 0.5,
                 temperature = 0.01
                 ) -> None:
        super().__init__()
        
        self.symbolic_base_dic = symbolic_base
        self.initialization = initialization
        self.input_channel = input_channel
        self.output_channel = output_channel # 暂时没用
        self.bias = bias
        self.device = device
        self.scale = scale
        self.skip_connect = skip_connect
        self.kernel_size = kernel_size
        self.stride = stride
        self.amount =  amount
        self.set_t(temperature)
        self.__init_layers__()
        

    def __init_layers__(self):
        self.func = [self.symbolic_base_dic['torch'][f] for f in self.symbolic_base_dic['torch']] # 取出dic中的函数
        self.symbolic_base = self.func * self.scale
        self.output_channel = int(len(self.symbolic_base))
        self.function_arity = []
        self.layer_size = 0
        self.learnable_param = nn.ModuleList([])
        for i,f in enumerate(self.symbolic_base):
            if isinstance(f, convlutional_operator):
                self.conv_op = f
                self.learnable_param.append(self.conv_op)
                self.symbolic_base[i] = (lambda x : self.conv_op(x) )              
            arity = get_arity(self.symbolic_base[i])
            self.function_arity.append((self.symbolic_base[i],arity))
            self.layer_size += arity
 
       
        self.In = nn.InstanceNorm1d(self.input_channel)
        self.channel_conv = nn.Conv1d(self.input_channel,self.layer_size,kernel_size=1,stride=1,padding=0,bias=self.bias)
        if self.skip_connect:
            self.down_conv = nn.Conv1d(self.input_channel,self.output_channel,kernel_size=1,stride= 1,padding=0,bias=self.bias)
        self.moving_avg_layer = nn.AvgPool1d(kernel_size = self.kernel_size,
                                                   stride = self.stride,
                                                   padding=(self.kernel_size-1)//2)

        # self.CA = attentionBlock(channel=self.output_channel,reduction=self.output_channel//2,
        #                          kernel_size=self.kernel_size) # 哪个结构attention 开始还是最后
    def weight_operator(self):
        pass
    
    def get_weight(self):
        return self.projection.weight.cpu().detach().numpy()
    def set_t(self,t):
        self.temperature = t
    def get_Matrix(self,x):
        '''
        x should be symbol
        '''

        self.sympy_func = [self.symbolic_base_dic['sympy'][f] for f in self.symbolic_base_dic['sympy']]
        self.sympy_symbolic_base = self.sympy_func * self.scale
        self.sympy_function_arity = []
        for f in self.sympy_symbolic_base:
            arity = get_arity(f)
            self.sympy_function_arity.append((f,arity))
      
        
        # 
        IN_layer = Matrix(x) # IN_layer = Matrix([Function('IN')(x_) for x_ in x])
        
        # IN_layer = IN_layer*len(x)
        
        # argmax weight
        layer_weight = prune_matrix(self.channel_conv,amount = self.amount)

        
        # channel
        
        args = layer_weight * IN_layer
        
        args_idx, signal = 0, []
        for i, (f, arity) in enumerate(self.sympy_function_arity):
            arg = args[args_idx: args_idx + arity]
            signal.append(f(*arg))
            args_idx = args_idx + arity
        signal = Matrix(signal)
        if self.skip_connect:
            # down_conv matrix
            layer_weight = prune_matrix(self.down_conv,amount = self.amount)
            signal_skip = layer_weight * IN_layer
            # signal = hadamard_product(signal,signal_skip) + signal_skip
            signal += signal_skip
        pooling = [s for s in signal] # pooling = [Function('pool')(s) for s in signal] # * 代表解开矩阵
        return pooling
      
        
        

        
    def forward(self,x):
        avg_x = self.moving_avg_layer(x)
        
        # normed_x = self.In(avg_x)
        normed_x = avg_x
        self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=0) # 权重蒸馏 , dim=0
        # self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=1) # 权重蒸馏 , dim=1
        multi_channel_x = self.channel_conv(normed_x) # 相当于针对通道的全连接层
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity
        if self.skip_connect:
            output += self.down_conv(normed_x) 
        return output
class neural_symbolic_regression(neural_symbolc_base):
    def __init__(self, symbolic_base=None, initialization=None, input_channel=1, output_channel=None, bias=False, device='cuda',
                 scale=1,kernel_size = None,stride = None, skip_connect=True,amount = 0,temperature = 0.01) -> None:
        super().__init__(symbolic_base, initialization, input_channel, output_channel,
                         bias, device, scale,kernel_size,stride,skip_connect,amount = amount,temperature = temperature)
    def forward(self,x):
        self.channel_conv.weight.data = F.softmax((1.0 / self.temperature) * self.channel_conv.weight.data, dim=0) # 权重蒸馏 , dim=0    
        multi_channel_x = self.channel_conv(x)
        
        args_idx = 0
        for i, (f, arity) in enumerate(self.function_arity):
            channel_x = multi_channel_x[:,args_idx: (args_idx + arity)]
            img = f(*torch.split(channel_x, 1, dim=1)) # split input + opearator + squeeze
            output = torch.cat((output, img), dim = 1) if i else  img
            args_idx += arity
        if self.skip_connect:    
            output += self.down_conv(x)
        return output        
    def get_Matrix(self,x):
        '''
        x should be symbol
        '''

        self.sympy_func = [self.symbolic_base_dic['sympy'][f] for f in self.symbolic_base_dic['sympy']]
        self.sympy_symbolic_base = self.sympy_func * self.scale
        self.sympy_function_arity = []
        for f in self.sympy_symbolic_base:
            arity = get_arity(f)
            self.sympy_function_arity.append((f,arity))
        layer_weight = prune_matrix(self.channel_conv,amount = self.amount)
        
        # channel
        
        args = layer_weight * Matrix(x)
        
        args_idx, signal = 0, []
        for i, (f, arity) in enumerate(self.sympy_function_arity):
            arg = args[args_idx: args_idx + arity]
            signal.append(f(*arg))
            args_idx = args_idx + arity
        signal = Matrix(signal)
        if self.skip_connect:
            # down_conv matrix
            layer_weight = prune_matrix(self.down_conv,amount = self.amount)
            signal_skip = layer_weight * Matrix(x)
            signal = signal + signal_skip
        pooling = [s for s in signal] # * 代表解开矩阵
        return pooling        

# %%          
class basic_model(nn.Module):
    def __init__(self,
                 input_channel = 1,
                 bias = False,
                 symbolic_bases = None,
                 scale = [1],
                 skip_connect = True,
                 down_sampling_kernel = None,
                 down_sampling_stride = 2,
                 num_class = 10,
                 device = 'cuda',
                 amount = 0.5,
                 temperature = 0.01
                 ) -> None:
        super().__init__()
        '''
        scale and symbolic_basede should be match
        '''     
        self.input_channel = input_channel
        self.bias = bias
        self.skip_connect = skip_connect
        self.amount = amount
        self.temperature = temperature
        self.scale = scale
        self.symbolic_bases = symbolic_bases
        self.down_sampling_kernel = down_sampling_kernel
        self.down_sampling_stride = down_sampling_stride
        self.device = device

        # signal processing layer
        self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = self.symbolic_bases,
                       input_channel= self.input_channel,
                       layer_type = 'transform')
        

        final_dim = self.symbolic_transform_layer[-1].output_channel
        
 
        # logical regression layer
        self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases[0]], # 1 layer
                       input_channel= final_dim,
                       layer_type = 'regression')
        
        final_dim = self.symbolic_regression_layer[-1].output_channel
        

        # Linear conbination
        self.regression_layer = nn.Linear(final_dim,num_class,bias = bias)
        
        self.to(self.device)
    def __make_layer__(self,
                       symbolic_bases,
                       input_channel = 1,
                       layer_type = 'transform' # 'regression'
                       ):            
        layers = []
        layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):                            
            next_channel = layer.output_channel if i else input_channel
            
            layer = layer_selection( symbolic_base = symbolic_base,
                 initialization = None,
                 input_channel = next_channel, 
                 output_channel = None,
                 bias = self.bias,
                 device = self.device,
                 scale = self.scale[i],
                 kernel_size = self.down_sampling_kernel[i],
                 stride = self.down_sampling_stride[i],
                 skip_connect = self.skip_connect,
                 amount = self.amount,
                 temperature = self.temperature)            
            layers.append(layer)

                
        return nn.ModuleList(layers)
    
    def norm(self,x):
        mean = x.mean(dim = 1,keepdim = True)
        std = x.std(dim = 1,keepdim = True)
        out = (x-mean)/(std + 1e-10)
        return out
        
    def forward(self,x):
        
        for layer in self.symbolic_transform_layer:
            x = layer(x)  

        self.feature = x.mean(dim=-1, keepdim = True) # TODO symbolize the statist feature
        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        x = x.squeeze()
        x = self.regression_layer(x)
        return x
class DEN(basic_model):
    '''
    individually add set the expert list and logical list, 
    whereb asic_model aims to the same operator
    '''
    def __init__(self, input_channel=1, bias=False, symbolic_bases=None, scale=[1],
     skip_connect=True, down_sampling_kernel=None, down_sampling_stride=2, num_class=10, device='cuda', amount=0.5,temperature = 0.01,
     expert_list = None, logic_list =None) -> None:
        super().__init__(input_channel, bias, symbolic_bases, scale,
                         skip_connect, down_sampling_kernel, down_sampling_stride, num_class, device, amount,temperature)

        self.expert_list = expert_list
        self.logic_list = logic_list
        self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = expert_list,
                       input_channel= self.input_channel,
                       layer_type = 'transform')
        final_dim = self.symbolic_transform_layer[-1].output_channel
        self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = [logic_list[0]], # 1 layer
                       input_channel= final_dim,
                       layer_type = 'regression')
        
        final_dim = self.symbolic_regression_layer[-1].output_channel
        self.regression_layer = nn.Linear(final_dim,num_class,bias = bias)
        
        self.to(self.device)
    def __make_layer__(self,
                       symbolic_bases,
                       input_channel = 1,
                       layer_type = 'transform' # 'regression'
                       ):            
        layers = []
        layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
        for i,symbolic_base in enumerate(symbolic_bases):                            
            next_channel = layer.output_channel if i else input_channel
            
            layer = layer_selection( symbolic_base = symbolic_base,
                 initialization = None,
                 input_channel = next_channel, 
                 output_channel = None,
                 bias = self.bias,
                 device = self.device,
                 scale = self.scale[i],
                 kernel_size = self.down_sampling_kernel[i],
                 stride = self.down_sampling_stride[i],
                 skip_connect = self.skip_connect,
                 amount = self.amount,
                 temperature = self.temperature)            
            layers.append(layer)

                
        return nn.ModuleList(layers)
    def forward(self,x):

        for layer in self.symbolic_transform_layer:
            x = layer(x)  
        self.signal = x
        self.feature = self.kurtosis(x)
        x = self.feature
        for layer in self.symbolic_regression_layer:
            x = self.norm(x)
            x = layer(x)
            
        self.logic_x = x.squeeze()
        x = self.regression_layer(self.logic_x)

        return x
    def kurtosis(self,x):
        # TODO symbolize it
        cen_order4 = torch.pow(x-x.mean(dim=-1, keepdim = True), 4).mean(dim=-1, keepdim = True)
        var_2 = torch.pow(x-x.mean(dim=-1, keepdim = True), 2).mean(dim=-1, keepdim = True)**2
        return cen_order4/var_2 - 2

#%%    
if __name__ == '__main__':  
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    save_dir = './save_dir'
    create_dir(save_dir)
    summaryWriter = SummaryWriter(save_dir +"/test")
    x = torch.randn(32,1,1024).cuda()
    
    base = symbolic_base(['add','mul','sin','exp','idt','sig','tanh','pi','e']) # ['add','mul','sin','exp','idt','sig','tanh','pi','e']
    

#%% test DNSN
    expert_list = symbolic_base(['mul','fft','squ','Morlet','Laplace','order1_MA','order2_MA','order2_MA','order2_DF'])

    logic_list= symbolic_base(['imp','equ','neg','conj','disj','sconj','sdisj'])
    model = DEN(input_channel = 1,
                 bias = False,
                 symbolic_bases = [base,base,base],
                 scale = [4,4,4],
                 skip_connect = True,
                 down_sampling_kernel = [7,7,7],
                 down_sampling_stride = [2,2,2],
                 num_class = 3,
                 expert_list = [expert_list,expert_list,expert_list],
                 logic_list = [logic_list,logic_list]
                 )
    y = model(x)
 
            
    print(summary(model,(1,1024),device = "cuda"))
