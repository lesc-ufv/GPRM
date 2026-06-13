from src.python.server_utils import *
class Sender:
    @staticmethod
    def initialize_models(server_config , launcher_void_return):
        return None
    
    @staticmethod
    def mapzero_infer(server_config , predicted_values):
        return tensor_to_msgpack(predicted_values)

    @staticmethod
    def config_server(server_config , launcher_void_return):
        return None
    
    @staticmethod
    def pre_train_mapzero( server_config , launcher_return):
        return {"total_loss":launcher_return[0], "policy_loss":launcher_return[1], "value_loss":launcher_return[2],
                "lr":launcher_return[3], "grad_norm": launcher_return[4]}
    
    @staticmethod
    def save_model( server_config , launcher_void_return):
        return None

    @staticmethod
    def load_pre_trained_model(server_config, launcher_void_return):
        return None
    

    
    @staticmethod
    def write_tensorboard( server_config, launcher_void_return):
        return None
    
    @staticmethod
    def finish_server(server_config, none_return):
        return None

    