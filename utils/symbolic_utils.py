from sympy import *
import torch.nn as nn
import torch.nn.utils.prune as prune
from inspect import signature
import torch.nn.functional as F
import torch
import sys
sys.path.append("../")
from model.symbolic_base import * 

import numpy as np
import pandas as pd
KEEPNUM = 4 # 保留位数
def prune_matrix(module, pruning_method = prune.l1_unstructured,amount = 0):
    '''
    input : module to be pruned
    output : sympy Matrix
    pruning_method : https://pytorch.org/docs/stable/search.html?q=torch.nn.utils.prune&check_keywords=yes&area=default#
    '''
    module.weight.data = F.softmax(module.weight.data, dim=0)
    if amount != 0:
        pruning_method(module,name="weight", amount = amount) # 裁剪
        prune.remove(module, 'weight')
    layer_weight = torch.round(module.weight,decimals = KEEPNUM*KEEPNUM) 
    layer_weight = layer_weight.detach()
    layer_weight = Matrix(layer_weight.squeeze(-1).cpu().numpy())
    
    return layer_weight
def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})
def element_mul(x,y):
    assert x.shape==y.shape
    out = [x[i] * y[i] for i in range(x.shape[0]*x.shape[1])] # [a*b for (a,b) in (x,y)]
    return Matrix(out).reshape(x.shape[0],x.shape[1])

def get_model_equation(model,
                       input_dim = 1,
                       output_dim = 10,
                       save_dir = './',
                       amount = 0.5
                       ):
    def output_symbol(signals, name = 'f'):
        dic = {}
        dic['sympy'] = []
        dic['latex'] = [] 

        for i,signal in enumerate(signals): 
            signal = round_expr(signal,KEEPNUM-1) # 保留两位
            dic['sympy'].append(f'{name}_{i+1}={signal}')
            dic['latex'].append(f'{name}_{i+1}={latex(signal)}') 
            print(f'learned {name}_{i} ====>> {signal}')     
        signals = symbols([name + '_' + str(i+1) for i in range(len(signals))]) # 用f来替换 学到的特征，便于打印
        pd.DataFrame(dic).to_csv(save_dir + f'/{amount}_{name}.csv')
        
        
        return signals
    signals = symbols(['x'])
    # signals = symbols(['X_' + str(i) for i in range(input_dim)])
    # constants = symbols([LATEX_CONSTANTS[constant][1:][:-1] for constant in model.constants])
    outputs = symbols(['Y_' + str(i+1) for i in range(output_dim)])
    


    # 初始化保存字典
    # feature_dic = {}  聚合了
    # feature_dic['sympy'] = []
    # feature_dic['latex'] = []
    
    # logic_dic = {}
    # logic_dic['sympy'] = []
    # logic_dic['latex'] = []  
    
    # output_dic = {}
    # output_dic['sympy'] = []
    # output_dic['latex'] = []    
   
    scale_length = len(model.scale)
    layers = []
    
    # 遍历网络，找到层
    for i, module in enumerate(model.children()):
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.AvgPool1d):
            layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]
    # 特征层取表达式
    for i,layer in enumerate(layers[0:scale_length]):
        signals = layer.get_Matrix(signals)
        
        matrix = pd.DataFrame(np.array(signals))
        matrix.to_csv(save_dir + f'/layer{i}.csv')
    
    # 计算太慢，直接
    # i,L = symbols('i L',integer=True)
    # signals = [sympy.summation(x_,(i,1,L)) for x_ in signals] # 写不到积分，把L看成求和符号吧
    
    # 化简
    # signals = [simplify(signal) for signal in signals] # 我们只分析稀疏的情况

    # 用 f 代替提取的特征
    
    ###################### todo  feature也加入
    signals = output_symbol(signals,name = 'f')
    # 符号回归层    
    for i,layer in enumerate(layers[scale_length:]): # 1层
        if not isinstance(layer, nn.Linear) :
            signals = layer.get_Matrix(signals)
            matrix = pd.DataFrame(np.array(signals))
            matrix.to_csv(save_dir + f'/srlayer{i}.csv')
    signals = output_symbol(signals,name = '\\tilde{y}')
    # for signal in signals:  
    #     signal = round_expr(signal,KEEPNUM-1) 
    #     output_dic['sympy'].append(signal)
    #     output_dic['latex'].append(latex(signal))         
      
    # signals = symbols(['\sigma_' + str(i+1) for i in range(len(signals))])   # 是否把logic 抽象出来                            
    # signals = [simplify(signal) for signal in signals]
        
    # 线性组合层    
    signals = prune_matrix(layers[-1],amount= 0.5) * Matrix(signals) # 全连接只prune 0.5，本来是按照给定值prune
    # 化简
    # signals = [simplify(signal) for signal in signals]
    signals = output_symbol(signals,name = 'y')

    
    return signals

def get_arity(f):
    return len(signature(f).parameters)

if __name__ == '__main__':  
    from torchsummary import summary
    from model.symbolic_layer import basic_model
    base = symbolic_base(['add','mul','sin','pi','e'])
    model = basic_model(input_channel = 1,
                 bias = False,
                 symbolic_bases = [base,base,base],
                #  symbolic_bases_4reg = None,
                 scale = [4,4,4],
                 skip_connect = True,
                 down_sampling_kernel = [8,8,8],
                 down_sampling_stride = [2,2,2],
                 num_class = 4,)
    x = torch.randn(32,1,1024).cuda()
    y = model(x)
    eq = get_model_equation(model,
                       input_dim = 1,
                       output_dim = 10,
                       save_dir = './')

    print(summary(model,(1,1024),device = "cuda"))