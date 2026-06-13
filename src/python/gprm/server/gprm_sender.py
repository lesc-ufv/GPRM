import os
from src.python.server_utils import *
class GPRMSender:
    @staticmethod
    def config_server(server_config, launcher_void_return):
        return None
    @staticmethod
    def initialize_models(server_config,receiver_return):
        return None
    @staticmethod
    def smartmap_infer(server_config , predicted_values):
        return tensor_to_msgpack(predicted_values)
    
    @staticmethod
    def ppo_training( server_config , launcher_return):
        return {"total_loss":launcher_return[0], "value_loss":launcher_return[1], "policy_loss":launcher_return[2],
                "entropy_loss": launcher_return[3], "kl_loss": launcher_return[4], "kl_coeff": launcher_return[5], 
                "total_norm":launcher_return[6], "lr": launcher_return[7], "entropy_coef": launcher_return[8]}
    @staticmethod
    def write_tensorboard( server_config, launcher_void_return):
        return None
    

    @staticmethod 
    def finish_server(server_config,json):
        return None
    @staticmethod
    def save_model(server_config, json):
        return None

    @staticmethod
    def update_old_model(server_config,json):
        return None
    @staticmethod
    def gen_laplacian_embeddings(server_config, launcher_return):
        return tensor_to_msgpack(launcher_return)

