import os
import yaml
from argparse import ArgumentParser,Namespace
import pandas as pd
import torch

def create_dir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
def setup_config(config_dir,sweep = False):
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f.read())

    # parser = ArgumentParser()
    # parser.add_argument("--backend", default=None, type=str)
    # for k, v in config.items():
    #     if isinstance(v, bool):
    #         parser.add_argument(f"--{k}", action="store_true")
    #     else:
    #         parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # args = parser.parse_args()
    if sweep:        
        setting = 'sweep'        
    else:
        symbolic_base_list = config['symbolic_base_list']
        scale = config['scale']
        name = config['name']
        setting = f'_name:{name}'  
    # setting = 'test3'
    config['setting'] = setting

    return Namespace(**config)

class loger():
    def __init__(self,head_list,save_dir='./save_dir/',save_name='Default'):
        self.train_his={}

        self.save_dir = save_dir
        self.head_list = head_list
        self.save_name = save_name
        for i in self.head_list:
            self.train_his[i]=[]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)     
            
    def add(self,value_list):       
        assert len(self.head_list)==len(value_list)
        
        for i,title in enumerate(self.head_list):
            if torch.is_tensor(value_list[i]):
                self.train_his[title].append(value_list[i].item())
            else:
                self.train_his[title].append(value_list[i])
                
    def export_csv(self):
        data_df = pd.DataFrame(self.train_his)

        data_df.to_csv(self.save_dir+self.save_name+'.csv',index=False,header=True)