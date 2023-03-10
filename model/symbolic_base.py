import sympy
from sympy import Function
import math
import torch # .nn as nn
import torch.nn as nn
import torch.nn.functional as F
from sympy.stats import Normal,density
from sympy.abc import W

KERNEL_SIZE = 49 
FRE = 10 
DEVICE = 'cuda'
STRIDE = 1

class Swish(nn.Module):
    # neural operator can also be adopted but with little interpretability 
	def __init(self,inplace=True):
		super(Swish,self).__init__()
		self.inplace=inplace
	def forward(self,x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x*torch.sigmoid(x)

def Morlet(t, a = pow(math.pi, 0.25),f = FRE):
    w = 2 * math.pi * f    
    y = a * torch.exp(-torch.pow(t, 2) / 2) * torch.cos(w * t)
    return y

def Laplace(t,a = 0.08,ep = 0.03,tal = 0.1,f = FRE):
    w = 2 * math.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = a * torch.exp((-ep / (torch.sqrt(q))) * (w * (t - tal))) * (-torch.sin(w * (t - tal)))
    return y

def chirplet(t,alpha,f):
    return abs(torch.exp(-0.5 * (t / 10)**2) * torch.exp(-1j * 2 * torch.pi * (alpha / 2 * t**2 + f * t)))

class convlutional_operator(nn.Module):
    '''
    conv_sin
    conv_exp
    Morlet
    Laplace
    chirplet
    '''
    def __init__(self, kernel_op = 'conv_sin',
                 dim = 1, stride = STRIDE,
                 kernel_size = KERNEL_SIZE,device = 'cuda',
                 ) -> None:
        super().__init__()

        self.affline = nn.InstanceNorm1d(num_features=dim,affine=True).to(device) 
        # TODO set parameter updation idividually rather than affiline
        op_dic = {'conv_sin':torch.sin,
                  'conv_sin2':lambda x: torch.sin(x**2),
                  'conv_exp':torch.exp,
                  'conv_exp2':lambda x: torch.exp(x**2),
                  'Morlet':Morlet,
                  'Laplace':Laplace,
                  'chirplet':chirplet}
        self.op = op_dic[kernel_op]
        self.stride = stride
        self.t = torch.linspace(-math.pi/2,math.pi/2, kernel_size).view(1,1,kernel_size).to(device) # cuda 
        self.kernel_size = kernel_size
        
    def forward(self,x):
        # length = x.shape[-1]
        self.aff_t = self.affline(self.t)
        self.weight = self.op(self.aff_t)
        conv = F.conv1d(x,self.weight, stride=self.stride, padding=(self.kernel_size-1)//2, dilation=1, groups=1)      

        return conv

class signal_filter_(nn.Module):
    '''
    order1_MA
    order2_MA
    order1_DF
    order2_DF
    '''
    def __init__(self, kernel_op = 'order1_MA',
                 dim = 1, stride = STRIDE,
                 kernel_size = KERNEL_SIZE,device = 'cuda',
                 ) -> None:
        super().__init__()
        op_dic = {'order1_MA':torch.Tensor([0.5,0,0.5]), #
                  'order2_MA':torch.Tensor([1/3,1/3,1/3]),
                  'order1_DF':torch.Tensor([-1,0,1]),
                  'order2_DF':torch.Tensor([-1,2,-1])}
        self.weight = op_dic[kernel_op].view(1,1,-1).to(device)
        self.stride = stride
        self.kernel_size = self.weight.shape[-1]       
    def forward(self,x):
        conv = F.conv1d(x, self.weight, stride=self.stride, padding=(self.kernel_size-1)//2, dilation=1, groups=1)
        return conv    

ONE = torch.Tensor([1]).cuda()
ZERO = torch.Tensor([0]).cuda()

def generalized_softmax(x,y, alpha = 20):
    numerator = x * torch.exp(alpha * x) + y * torch.exp(alpha * y)
    denominator = torch.exp(alpha * x) + torch.exp(alpha * y)
    return numerator / denominator 
def generalized_softmin(x,y, alpha = 20):
    return -generalized_softmax(-x,-y, alpha = alpha)

def implication(x, y):
    return generalized_softmin(ONE, ONE - x + y)

def equivalence(x, y):
    return ONE - torch.abs(x - y)

def negation(x):
    return ONE - x

def weak_conjunction(x, y):
    return generalized_softmin(x, y)

def weak_disjunction(x, y):
    return generalized_softmax(x, y)

def strong_conjunction(x, y):
    return generalized_softmax(ZERO, x + y - 1)

def strong_disjunction(x, y):
    return generalized_softmin(ONE, x + y)

#

def symbolic_base( given_set = ['add','mul','sin','exp','idt','sig'], fre = FRE):  # 
    
    ## few variable
    
    
    add = ['add', (lambda x, y: x + y), (lambda x, y: x + y), '$+$']
    mul = ['mul', (lambda x, y: x * y), (lambda x, y: x * y), '$\\times$']
    div = ['div', (lambda x, y: x / y), (lambda x, y: x / y), '$\div$']
    squ = ['squ', (lambda x: x**2), (lambda x: x**2),'$x^2$']
    sin = ['sin', (lambda x: torch.sin( fre * x)), (lambda x: sympy.sin(x)), '$\sin$']
    arcsin = ['arcsin', (lambda x: torch.arcsin( fre * x)), (lambda x: sympy.asin(x)), '$\\asin$']
    idt = ['idt', (lambda x: x), (lambda x: x), '$\mathbf{I}$']
    sig = ['sig', lambda x: torch.sigmoid(fre * x), lambda x: 1 / (1 + sympy.exp(x)), "$\sigma$"]
    X = Normal("X", 0, 1)
    exp = ['exp',lambda x: torch.exp( - fre* (x) **2),lambda x: density(X)(x) * sympy.sqrt(2*sympy.pi), "$e^{\left(-\\frac{x^{2}}{2}\\right)$" ]
    log = ['log',lambda x: torch.log(x),lambda x: sympy.log(x), "$\log$"]
    tanh = ['tanh',lambda x: torch.tanh(x),lambda x: sympy.tanh(x), "$\tanh$"]
    swish = ['swish',(lambda x: Swish()(x)), (lambda x: sympy.Function('swish')(x))]
    
    # constant
    
    pi             = ['pi',lambda x: x*math.pi,  lambda x: x*sympy.pi, "$\pi$"]
    e              = ['e',lambda x: x*math.e,  lambda x: x*sympy.E, "$e$"]
    
    
 

    # signal processing operator
    fft            = ['fft',lambda x: torch.abs(torch.fft.fft(x)),  lambda x: sympy.Function('fft')(x), "$fft$"]
    
    conv_sin      = ['conv_sin',convlutional_operator('conv_sin'), (lambda x: sympy.Function('conv_sin')(x)), "$*_{conv_sin}$"]
    conv_sin2      = ['conv_sin2',convlutional_operator('conv_sin2'), (lambda x: sympy.Function('conv_sin2')(x)), "$*_{conv_sin2}$"]
    conv_exp      = ['conv_exp',convlutional_operator('conv_exp'), (lambda x: sympy.Function('conv_exp')(x)), "$*_{conv_exp}$"]
    conv_exp2      = ['conv_exp2',convlutional_operator('conv_exp2'), (lambda x: sympy.Function('conv_exp2')(x)), "$*_{conv_exp2}$"]
    Morlet      = ['Morlet',convlutional_operator('Morlet'), (lambda x: sympy.Function('Morlet')(x)), "$*_{Morlet}$"]
    Laplace      = ['Laplace',convlutional_operator('Laplace'), (lambda x: sympy.Function('Laplace')(x)), "$*_{Laplace}$"]
    
    order1_MA      = ['order1_MA',(lambda x: signal_filter_('order1_MA')(x)), (lambda x: sympy.Function('MA_1')(x)), "$*_{MA_1}$"]
    order2_MA      = ['order2_MA',(lambda x: signal_filter_('order2_MA')(x)), (lambda x: sympy.Function('MA_2')(x)), "$*_{MA_2}$"]
    order1_DF      = ['order1_DF',(lambda x: signal_filter_('order1_DF')(x)), (lambda x: sympy.Function('DF_1')(x)), "$*_{DF_1}$"]
    order2_DF      = ['order2_DF',(lambda x: signal_filter_('order2_DF')(x)), (lambda x: sympy.Function('DF_2')(x)), "$*_{DF_2}$"]
    
    # logical operator
    
    imp = ['imp',(lambda x,y: implication(x,y)), (lambda x,y: sympy.Function('F_imp')(x,y))]  # F_→
    equ = ['equ',(lambda x,y: equivalence(x,y)), (lambda x,y: sympy.Function('F_equ')(x,y))] #F_↔
    neg = ['neg',(lambda x: negation(x)), (lambda x: sympy.Function('F_neg')(x))] # F_¬           
    conj = ['conj',(lambda x,y: weak_conjunction(x,y)), (lambda x,y: sympy.Function('F_conj')(x,y))] # 'F_∧'
    disj = ['disj',(lambda x,y: weak_disjunction(x,y)), (lambda x,y: sympy.Function('F_disj')(x,y))] # 'F_∨'
    sconj = ['sconj',(lambda x,y: strong_conjunction(x,y)), (lambda x,y: sympy.Function('F_sconj')(x,y))] # 'F_⨂'
    sdisj = ['sdisj',(lambda x,y: strong_disjunction(x,y)), (lambda x,y: sympy.Function('F_sdisj')(x,y))] # 'F_⨁'
    
    total_set = {
        'add':add,'mul':mul,'div':div,'sin':sin,'arcsin':arcsin,'idt':idt,'sig':sig,'exp':exp,'log':log,'tanh':tanh,'swish':swish,
        'pi':pi,'e':e,
        'fft':fft,'conv_sin':conv_sin,'conv_sin2':conv_sin2,'conv_exp':conv_exp,'conv_exp2':conv_exp2,'squ':squ,'Morlet':Morlet,'Laplace':Laplace,
        'order1_MA':order1_MA,'order2_MA':order2_MA,'order1_DF':order1_DF,'order2_DF':order2_DF,
        'imp':imp,'equ':equ,'neg':neg,'conj':conj,'disj':disj,'sconj':sconj,'sdisj':sdisj
    
    }
    
    torch_base = { fuc: total_set[fuc][1] for fuc in given_set}
    sympy_base = { fuc: total_set[fuc][2] for fuc in given_set}
    # latex_base = { fuc: total_set[fuc][3] for fuc in given_set}
    
    return {
        'torch':torch_base,
        'sympy':sympy_base,
    }
def feature_base():
    '''
    for f-eql
    '''
    ori_order1_2 = ["OM1_2O",(lambda x : torch.sqrt(torch.abs(x))),
                        (lambda x: Function('OM1_2O')(x)),
                        '$OM1_2O$']
    ori_order2_2 = ["OM2_2O",(lambda x : torch.sqrt(torch.pow(x, 2))),
                        (lambda x: Function('OM2_2O')(x)),
                        '$OM2_2O$']
    ori_order1 = ["OM1O",(lambda x : x),
                    (lambda x: Function('OM1O')(x)),
                    '$OM1O$']
    ori_order2 = ["OM2O",(lambda x :  torch.pow(x, 2)),
                        (lambda x: Function('OM2O')(x)),
                        '$OM2O$']
    ori_order3 = ["OM3O",(lambda x :  torch.pow(x, 3)),
                        (lambda x: Function('OM3O')(x)),
                        '$OM3O$']
    ori_order4 = ["OM4O",(lambda x :  torch.pow(x, 4)),
                        (lambda x: Function('OM4O')(x)),
                        '$OM4O$']

    cen_order1_2 = ["CM1_2O",(lambda x : torch.sqrt(torch.abs(x-torch.mean(x)))),
                        (lambda x: Function('CM1_2O')(x)),
                        '$CM1_2O$']
    cen_order2_2 = ["CM2_2O",(lambda x : torch.sqrt(torch.pow(x-torch.mean(x), 2))),
                        (lambda x: Function('CM2_2O')(x)),
                        '$CM2_2O$']
    cen_order1 = ["CM1O",(lambda x : x-torch.mean(x)),
                        (lambda x: Function('CM1O')(x)), '$CM1O$']
    cen_order2 = ["CM2O",(lambda x : torch.pow(x-torch.mean(x), 2)),
                        (lambda x: Function('CM2O')(x)), '$CM2O$']
    cen_order3 = ["CM3O",(lambda x : torch.pow(x-torch.mean(x), 3)),
                        (lambda x: Function('CM3O')(x)), '$CM3O$']
    cen_order4 = ["CM4O",(lambda x : torch.pow(x-torch.mean(x), 4)),
                        (lambda x: Function('CM4O')(x)), '$CM4O$']
    total_set = {
        'ori_order1_2':ori_order1_2,
        'ori_order2_2':ori_order2_2,
        'ori_order1':ori_order1,
        'ori_order2':ori_order2,
        'ori_order3':ori_order3,
        'ori_order4':ori_order4,
        'cen_order1_2':cen_order1_2,
        'cen_order2_2':cen_order2_2,
        'cen_order1':cen_order1,
        'cen_order2':cen_order2,
        'cen_order3':cen_order3,
        'cen_order4':cen_order4,        
    }
    
    torch_base = { fuc: total_set[fuc][1] for fuc in total_set}
    sympy_base = { fuc: total_set[fuc][2] for fuc in total_set}

    
    return {
        'torch':torch_base,
        'sympy':sympy_base,
    } 


def features_extractor(x):
    # stastic features
    N = x.shape[0]  
    mean_x = x.mean()
    delta_x = x[1:] - x[:-1]
    delta_mean = delta_x.mean()
    max_x = x.max()
    min_x = x.min()
    abs_mean = x.abs().mean()
    std = torch.sqrt(((x - mean_x)**2).mean())
    rms = torch.sqrt((x**2).mean())
    variance = ((x - mean_x)**2).mean()
    crest_factor = max_x / rms
    clearance_factor = max_x / abs_mean
    kurtosis = ((x - mean_x)**4).mean() / (variance ** 2)
    skewness = ((x - mean_x)**3).mean() / (std ** 3)
    shape_factor = rms / abs_mean
    crest_factor_delta = torch.sqrt(delta_x.pow(2).mean()) / abs_mean
    kurtosis_delta = ((delta_x - delta_mean)**4).mean() / (((delta_x - delta_mean)**2).mean())**2
    return max_x, min_x, abs_mean, std, rms, variance, crest_factor, clearance_factor, kurtosis, skewness, shape_factor, crest_factor_delta, kurtosis_delta

#%%   
if __name__ == '__main__':
    base = symbolic_base(['fft','squ','Morlet','Laplace','order1_MA','order2_MA','order1_DF','order2_DF'])
    print(base)
        