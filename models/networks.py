import torch.nn as nn
import torch
from models.Stripformer import Stripformer

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer()
        # GPU が使えるか確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # モデルを CUDA に移動
        model_g.to(device)
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
