from src.python.models.mapzero import MapZero
from src.python.constants.server_constants import ServerConstants
import torch
class Receiver:


    
  
    
    @staticmethod
    def mapzero_infer(server_config, json, tensors):
        return tensors
    
    @staticmethod
    def config_server(server_config, json):
       return json
    
    @staticmethod
    def pre_train_mapzero(server_config, json, tensors):
       return {"model_inputs":tensors[0:8],
               "actions":tensors[8],
               "target_values":tensors[9],
               "advantages":tensors[10]}
    
    @staticmethod
    def load_pre_trained_model(server_config, json):
        if server_config.model_id == 0:
            model = torch.load(server_config.checkpoint_pathname + server_config.checkpoint_name)
            optimizer = torch.optim.SGD(model.parameters(),5e-4)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return model,optimizer,lr_scheduler

    
    @staticmethod
    def write_tensorboard(server_config, json):
        return json

    @staticmethod 
    def finish_server(server_confi,json):
        return None
    @staticmethod
    def save_model(server_config, json):
        return None

