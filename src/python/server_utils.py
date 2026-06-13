from src.python.train_config import TrainConfig
from src.python.models.mapzero import MapZero
import torch
import msgpack
from src.python.constants.server_constants import ServerConstants
def initialize_model_by_id(id: int, args):
    if id == 0:
        return MapZero(*args)

def set_device(train_config: TrainConfig, device:str):
    train_config.device = device

def tensor_to_msgpack(tensors):
    tensor_specs = [] 
    tensor_messages = []  
    for i,tensor in enumerate(tensors):
        tensor = tensor.contiguous()  
        
        tensor_spec = {
            ServerConstants.SHAPE: tensor.shape,  
            ServerConstants.DTYPE: str(tensor.dtype),  
        }
        tensor_specs.append(tensor_spec)

        tensor_data = tensor.to('cpu').detach().numpy().tobytes()
        tensor_messages.append(tensor_data)
    packed_data = {
        ServerConstants.TENSORS: tensor_specs,  
        ServerConstants.DATA_K: tensor_messages, 
    }

    return packed_data