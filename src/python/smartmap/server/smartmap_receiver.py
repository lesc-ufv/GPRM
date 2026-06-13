import math
import os
from src.python.models.smartmap import SmartMap
from src.python.constants.server_constants import ServerConstants
import torch

class SmartMapReceiver:
    @staticmethod
    def config_server(server_config, json):
        server_config.update(json)
        return None
    @staticmethod
    def update_old_model(server_config,json):
        return None

    @staticmethod
    def smartmap_infer(server_config, json,tensors):
        tensors[5] = tensors[5].unsqueeze(0)
        return tensors
    
    @staticmethod
    def initialize_models(server_config,json):
        dim_dfg_feat = json[ServerConstants.IN_DIM_FEAT_DFG]
        dim_cgra_feat = json[ServerConstants.IN_DIM_FEAT_CGRA]
        out_dim = json[ServerConstants.OUT_DIM]
        n_heads = json[ServerConstants.N_HEADS]
        model = SmartMap(dim_dfg_feat,dim_cgra_feat,out_dim,n_heads)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p,gain=math.sqrt(2))
        optimizer = torch.optim.AdamW(model.parameters(),json[ServerConstants.LEARNING_RATE])
        return model, optimizer

    @staticmethod
    def ppo_training(server_config,json, tensors):
        return {"model_inputs":tensors[0:9],
               "target_values":tensors[9],
               "actions_indices":tensors[10],
               "advantages":tensors[11],
               "cur_epoch": json["cur_epoch"],
               ServerConstants.ENTROPY_COEF: json[ServerConstants.ENTROPY_COEF]}
    @staticmethod
    def write_tensorboard(server_config, json):
        return json
    @staticmethod
    def save_model( server_config, json):
        return None
    @staticmethod
    def finish_server(server_config, none_return):
        return None