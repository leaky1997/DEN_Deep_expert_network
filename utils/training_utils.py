#%%

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import time
from torchmetrics import Accuracy,F1Score,ConfusionMatrix
#%%
def setup_seed(seed):
    '''
    设置种子
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
     
class EarlyStopping:
    '''
    早停
    '''
    def __init__(self, patience=7, verbose=False, delta=0, scheduler = None, sparse = ''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.scheduler = scheduler
        self.sparse = sparse
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.scheduler is not None: # 到一般开始调整学习率
                self.scheduler.step(val_loss)  # 剪枝之后都是微调，因此学习率还会越来越小
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''
        path 保存路径
        sparse 是否读取sparse 的pt
        '''
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+ self.sparse +'checkpoint.pth')
        self.val_loss_min = val_loss
        
    def load_checkpoint(self,model,path):    
        return model.load_state_dict(torch.load(path+'/'+ self.sparse+ 'checkpoint.pth'))
    
    def reset(self,sparse):
        self.early_stop = False
        self.sparse = sparse
        self.counter = 0
        self.best_score = None # reset early_stoping        
                
def set_requires_grad(model, requires_grad=True):
    '''
    冻结梯度
    '''
    for param in model.parameters():
        param.requires_grad = requires_grad
        
def adjust_learning_rate():
    pass

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# class timer():
#     '''
#     记录时间
#     '''
#     def __init__(self):
#         self.start_time = time.time()
#     def time_cost(self):
#         self.record_time = time.time()
#         self.cost = self.record_time - self.start_time
#         self.start_time = time.time()
#         return self.cost
#%%
class metric_recorder():
    def __init__(self,metric_list, num_classes =10, top_k = 1, device = 'cuda') -> None:
        '''
        使用各种各样的metric 评估模型
        多分类则需要继承这个库
        '''
        self.metric_predefined = {'acc':Accuracy(top_k=top_k).to(device),
            'f1':F1Score(num_classes = num_classes).to(device),
            'cm':ConfusionMatrix(num_classes = num_classes).to(device)}
        # MulticlassAccuracy https://torchmetrics.readthedocs.io/en/latest/classification/accuracy.html?highlight=accuracy
        
        self.output_dic = {}
        self.metric_list = metric_list
        self.num_classes = num_classes
        for met in metric_list:
            assert met in self.metric_predefined
            self.output_dic[met] = []
    def update(self,y_pre,y):
        # y_pre = nn.functional.one_hot(torch.argmax(y_pre,dim=1),num_classes=self.num_classes) # 取索引，然后保存成onehot形式  for 多分类
        # y = y.type_as(y_pre)  
        y = torch.argmax(y,dim=1)     
        for met in self.metric_list: 
            metric_value = self.metric_predefined[met](y_pre,y)
            self.output_dic[met].append(metric_value.cpu().numpy())
    def mean(self):
        for met in self.metric_list:
            self.output_dic[met] = np.mean(self.output_dic[met])          
        return self.output_dic
    def sum(self):
        for met in self.metric_list:
            self.output_dic[met] = np.sum(self.output_dic[met])            
        return self.output_dic 
class cm_recorder():
    def __init__(self, num_classes =10, top_k = 1, device = 'cuda') -> None:
        '''
        使用各种各样的metric 评估模型
        多分类则需要继承这个库
        '''

        # MulticlassAccuracy https://torchmetrics.readthedocs.io/en/latest/classification/accuracy.html?highlight=accuracy
        
        self.output = 0
        self.metric = ConfusionMatrix(num_classes = num_classes).to(device)
        self.num_classes = num_classes

    def update(self,y_pre,y):
        # y_pre = nn.functional.one_hot(torch.argmax(y_pre,dim=1),num_classes=self.num_classes) # 取索引，然后保存成onehot形式  for 多分类
        # y = y.type_as(y_pre)  
        y = torch.argmax(y,dim=1)     
        self.output += self.metric(y_pre,y)

class loss_recorder():
    def __init__(self,loss_list, num_classes =10) -> None:
        '''
        使用各种各样的 loss 优化模型,但是好像不常用
        '''        
        self.loss_predefined = {'CE':nn.CrossEntropyLoss(),
            'mse':nn.MSELoss(),
            }
                
        self.output_dic = {}
        self.loss_list = loss_list
        for met in self.loss_list:
            assert met in self.loss_predefined
            self.output_dic[met] = []
    def update(self,y,y_pre):
        for met in self.loss_list:
            metric_value = self.loss_predefined[met](y,y_pre)
            self.output_dic[met].append(metric_value)
    def mean(self):
        for met in self.loss_list:
            self.output_dic[met] = torch.mean(self.output_dic[met])            
        return self.output_dic
    def sum(self):
        for met in self.loss_list:
            self.output_dic[met] = torch.sum(self.output_dic[met])            
        return self.output_dic    
    
def loop_iterable(iterable):
    while True:
        yield from iterable        
if __name__ == "__main__":
    recoder = metric_recorder(['cm'],device = 'cpu')
    pre = torch.randn(32,10)
    y = nn.functional.one_hot(torch.argmax(torch.randn(32,10),dim=1),num_classes=10)
    recoder.update(pre,y)
    recoder.update(pre,y)
    recoder
    print(recoder.sum())
    
#%%    discard
    pre = torch.randn(32,10)      
    y = nn.functional.one_hot(torch.argmax(torch.randn(32,10),dim=1),num_classes=10)
    cm = cm_recorder(device = 'cpu')
    cm.update(pre,y)
    cm.update(pre,y)
    print(cm.output)
     