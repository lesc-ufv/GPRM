from copy import deepcopy
from torch.distributions.categorical import Categorical
import torch
from src.python.models.transmap.transmap import TransMap
from src.python.models.smartmap import SmartMap
from src.python.models.gprm.gprm import GPRM
from src.python.laplacian import get_laplacian_emb
from src.python.models.mapzero import MapZero
from src.python.server_utils import tensor_to_msgpack
from src.python.constants.server_constants import ServerConstants
import os
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import json
import numpy as np
import random
import time
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import shutil
import threading
from typing import List


class ModelInputs:

    @staticmethod
    def _get_mapzero_keys():
        return [
            "in_dim_feat_dfg",
            "in_dim_feat_cgra",
            "out_dim",
            "n_heads",
            "number_of_spatial_PEs",
            "negative_slope"
        ]

    @staticmethod
    def _get_smartmap_keys():
        return [
            "in_dim_feat_dfg",
            "in_dim_feat_cgra",
            "out_dim",
            "n_heads",
            "negative_slope"
        ]

    @staticmethod
    def _get_gprm_keys():
        return [
            "lpe_dim",
            "vocab_len",
            "feat_out_dim",
            "out_dim",
            "ff_out_dim",
            "GIN_layers_list",
            "n_graph_embed_layers",
            "n_transformer_layers",
            "n_heads",
            "activation_func",
            "use_mlp",
            "dropout",
            "use_rms_norm",
            "use_norm_before_pred",
            "negative_slope",
            "max_len",
            "norm_initial_emb",
            "use_gin_emb",
            "use_spatio_temporal_info",
            "use_gnn",
            "use_transformer",
            "use_placement_order"
        ]

    @staticmethod
    def _get_gprm_keys():
        return [
        "lpe_dim",
        "vocab_len",
        "feat_out_dim",
        "out_dim",
        "ff_out_dim",
        "GIN_layers_list",
        "n_graph_embed_layers",
        "n_transformer_layers",
        "n_heads",
        "activation_func",
        "use_mlp",
        "dropout",
        "use_rms_norm",
        "use_norm_before_pred",
        "negative_slope",
        "max_len",
        "norm_initial_emb",
        "use_gin_emb",
        "use_mobility_info",
        "use_coord_info",
        "use_placement_order",
        "use_gnn",
        "use_transformer",
        "ignore_lpe",
        "use_scale"
    ]



import os
import time
import threading
from typing import List
import torch


import os
import time
import threading
from typing import List, Optional

import torch


class ChunkedSupervisedDatasetLoader:
    def __init__(self, dataset_dir: str, check_interval: float = 5.0):
        self.dataset_dir = dataset_dir
        self.check_interval = check_interval
        self.chunk_paths: List[str] = []
        self._lock = threading.RLock()  # FIX 5: RLock em vez de Lock para evitar deadlock reentrante
        self.cur_chunk_idx = 0
        self.cur_batch_idx = 0

        self.current_chunk: Optional[List] = None
        self.next_chunk: Optional[List] = None
        self._prefetch_thread: Optional[threading.Thread] = None

        self._update_chunk_list()

        assert len(self.chunk_paths) > 0, "No .pth chunks found"
        print("[INFO] Found chunks: ", self.chunk_paths)
        self.num_chunks = len(self.chunk_paths)

        self._load_chunk(self.cur_chunk_idx)
        self._prefetch_next()

        self._stop_monitor = False
        self._monitor_thread = threading.Thread(target=self._monitor_folder)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def _update_chunk_list(self):
        with self._lock:
            current_set = set(self.chunk_paths)
            new_chunks = sorted([
                os.path.join(self.dataset_dir, f)
                for f in os.listdir(self.dataset_dir)
                if f.endswith(".pth") and os.path.join(self.dataset_dir, f) not in current_set
            ])
            if new_chunks:
                self.chunk_paths.extend(new_chunks)
                self.num_chunks = len(self.chunk_paths)
                print(f"[INFO] Detected new chunks: {new_chunks}")

                if self.next_chunk is None:
                    # FIX 5: seguro chamar _prefetch_next aqui pois agora usamos RLock
                    self._prefetch_next()

    def _monitor_folder(self):
        while not self._stop_monitor:
            self._update_chunk_list()
            time.sleep(self.check_interval)

    def _load_chunk(self, idx: int):
        chunk_path = self.chunk_paths[idx]
        chunk_data = torch.load(chunk_path, map_location="cpu")

        with self._lock:
            self.current_chunk = chunk_data
            self.cur_batch_idx = 0

    def _prefetch_next(self):
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

        with self._lock:
            next_idx = (self.cur_chunk_idx + 1) % self.num_chunks
            next_chunk_path = self.chunk_paths[next_idx]

        def _load():
            try:
                # I/O fora do lock
                next_chunk_data = torch.load(next_chunk_path, map_location="cpu")
                with self._lock:
                    self.next_chunk = next_chunk_data
            except Exception as e:
                print(f"[ERROR] Prefetch failed: {e}")

        self._prefetch_thread = threading.Thread(target=_load)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()

    def next_batch(self):
        with self._lock:
            if self.cur_batch_idx >= len(self.current_chunk):
                self.cur_chunk_idx = (self.cur_chunk_idx + 1) % self.num_chunks

                if self.next_chunk is not None:
                    self.current_chunk = self.next_chunk
                    self.next_chunk = None
                    self.cur_batch_idx = 0
                    do_prefetch = True
                else:
                    do_prefetch = False

            else:
                do_prefetch = False

            if not do_prefetch and self.cur_batch_idx >= len(self.current_chunk):
                chunk_idx_to_load = self.cur_chunk_idx
                need_sync_load = True
            else:
                need_sync_load = False

            batch_idx = self.cur_batch_idx
            chunk_idx = self.cur_chunk_idx

        if need_sync_load:
            self._load_chunk(chunk_idx_to_load)
            self._prefetch_next()
            with self._lock:
                batch_idx = self.cur_batch_idx
                chunk_idx = self.cur_chunk_idx

        if do_prefetch:
            self._prefetch_next()

        with self._lock:
            batch = self.current_chunk[self.cur_batch_idx]
            self.cur_batch_idx += 1
            log_batch = self.cur_batch_idx
            log_chunk = self.cur_chunk_idx

        print(f"[INFO] Loaded batch {log_batch} from chunk {log_chunk}")
        return batch

    def state_dict(self):
        with self._lock:
            return {
                "cur_chunk_idx": self.cur_chunk_idx,
                "cur_batch_idx": self.cur_batch_idx,
                "dataset_dir": self.dataset_dir,
                "num_chunks": self.num_chunks,
                "chunk_paths": list(self.chunk_paths)
            }

    def load_state_dict(self, state: dict):
        assert state["dataset_dir"] == self.dataset_dir, "Dataset directory mismatch"

        with self._lock:
            self.cur_chunk_idx = state["cur_chunk_idx"]
            self.cur_batch_idx = state["cur_batch_idx"]
            self.chunk_paths = state.get("chunk_paths", self.chunk_paths)
            self.num_chunks = len(self.chunk_paths)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

        self._load_chunk(self.cur_chunk_idx)
        self._prefetch_next()

    def stop(self):
        self._stop_monitor = True
        if self._monitor_thread.is_alive():
            self._monitor_thread.join()

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

class Launcher:
    def __init__(self, max_checkpoints_to_save, grad_accumulate_steps):
        self.model: torch.nn.Module
        self.actor_optimizer : torch.optim.Optimizer
        self.critic_optimizer : torch.optim.Optimizer
        self.writer = None 
        self.last_priorities = None
        self.w_kl = None
        self.cur_step = None
        self.dataset = None
        self.trained_samples = None
        self.max_checkpoints_to_save = max_checkpoints_to_save
        self.last_saved_paths = [None for i in range(self.max_checkpoints_to_save)]
        self.num_saved_checkpoints = 0
        self.saved_blocks = 0
        self.supervised_loader: ChunkedSupervisedDatasetLoader = None
        self.grad_accumulate_steps = grad_accumulate_steps
    
    def save_block_of_supervised_data(self, server_config, json):
        assert self.dataset is not None and len(self.dataset) != 0, "No dataset to save"

        save_path = json["path_to_save_data"]

        os.makedirs(save_path, exist_ok=True)

        existing_files = [f for f in os.listdir(save_path) if f.endswith(".pth")]
        block_number = len(existing_files)

        torch.save(self.dataset, os.path.join(save_path, f"{block_number}.pth"))
        print(f"[INFO] Saved block {block_number} to {save_path}")

        self.dataset = []
    

    def get_last_priorities(self, server_config, json):
        if self.last_priorities is None:
            raise ValueError("No priorities available")
        temp_priorities = self.last_priorities
        self.last_priorities = None
        return tensor_to_msgpack(temp_priorities)
    
    def infer(self, server_config, unpacked_data, tensors):
        self.model.eval()
        init_time = time.time()
        if server_config["device"] == "cuda":
            tensors = [tensor.to("cuda") for tensor in tensors ]
        with torch.no_grad():
            values = self.model(*tensors)
        final_time = time.time()
        print("[INFO] batch size, time: ", len(values[0]), final_time - init_time)
        #torch.cuda.empty_cache()
        # time.sleep(1)
        return tensor_to_msgpack(values)
    
    def infer_with_attention_ret(self, server_config, unpacked_data, tensors):
        self.model.eval()
        init_time = time.time()
        if server_config["device"] == "cuda":
            tensors = [tensor.to("cuda") for tensor in tensors ]
        with torch.no_grad():
            values = self.model(*tensors)
        values.append(self.model.attn_outputs.to("cpu"))
        final_time = time.time()
        print("[INFO] batch size, time: ", len(values[0]), final_time - init_time)
        #torch.cuda.empty_cache()
        # time.sleep(1)
        return tensor_to_msgpack(values)

    
        
    @staticmethod
    def init_model(server_config):
        print("[INFO] Initializing model from scratch...")
        if server_config[ServerConstants.MODEL_ID] == 0:
            model_name = "MapZero"
            model_args = Launcher._get_mapzero_model_input_from_json(server_config)
            model = MapZero(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 1:
            model_name = "SmartMap"
            model_args = Launcher._get_smartmap_model_input_from_json(server_config)
            model = SmartMap(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 2:
            model_name = "GPRM"
            model_args = Launcher._get_gprm_input_from_dict(server_config)
            model = GPRM(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 3:
            model_name = "TransMap"
            model_args = Launcher._get_transmap_input_from_dict(server_config)
            model = TransMap(*model_args)
        else: 
            raise ValueError(f"ID {server_config[ServerConstants.MODEL_ID]} doesn't exist.")
        print(f"[INFO] {model_name} initialized with args :\n{model_args}")
        if server_config["use_xavier_uniform"]:
            print(f"[INFO] Initializing {model_name} with Xavier uniform")
            for p in model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
        
        return model

    def load_nn_only(self, server_config, load_path):
        json_config = Launcher.load_config(load_path+"../")

        gpu_is_available = torch.cuda.is_available()
       
        checkpoint = torch.load(load_path+ "checkpoint.pth",
                                weights_only=True,
                                map_location="cuda" if gpu_is_available else "cpu") 
     
        if json_config[ServerConstants.MODEL_ID] == 0:
            model_args = Launcher._get_mapzero_model_input_from_json(json_config)
            model = MapZero(*model_args)
            keys = ModelInputs._get_mapzero_keys()
        elif json_config[ServerConstants.MODEL_ID] == 1:
            model_args = Launcher._get_smartmap_model_input_from_json(json_config)
            model = SmartMap(*model_args)
            keys = ModelInputs._get_smartmap_keys()

        elif json_config[ServerConstants.MODEL_ID] == 2:
            model_args = Launcher._get_gprm_input_from_dict(json_config)
            model = GPRM(*model_args)
            keys = ModelInputs._get_gprm_keys()
        elif json_config[ServerConstants.MODEL_ID] == 3:
            model_args = Launcher._get_transmap_input_from_dict(json_config)
            model = TransMap(*model_args)
            keys = ModelInputs._get_transmap_keys()
        else:
            raise RuntimeError(f"Model not loaded - Model ID {json_config[ServerConstants.MODEL_ID]}")
        
        model = model.to(server_config["device"])
        model.load_state_dict(checkpoint["model_state"])
        for key in keys:
            server_config.update({key: json_config[key]})
        self.model = model
    
    @staticmethod
    def load_model(server_config, load_path):
        json_config = Launcher.load_config(load_path)
        load_checkpoint_path = server_config["load_checkpoint_path"]
        save_checkpoint_path = server_config["save_checkpoint_path"]
        server_config.update(json_config)
        if server_config['load_checkpoint_path'] == "":
            server_config["load_checkpoint_path"] = load_checkpoint_path
        if server_config["save_checkpoint_path"] == "":
            server_config["save_checkpoint_path"] = save_checkpoint_path
        specific_iter = False
        if "iter" in load_path:
            # get parent dir
            train_json_path = os.path.dirname(load_path) + "/"
            specific_iter = True
        else:
            train_json_path = load_path
        with open(train_json_path + "train_info.json", "r") as f:
            train_info = json.load(f)
            if specific_iter:
                iter = int(load_path.split("iter_")[-1].strip("/"))
            else:
                iter = train_info["final_iter"]
            cur_step = train_info["cur_step"]
            trained_samples = train_info["trained_samples"]     
            last_saved_paths = train_info["last_saved_paths"] 
            num_saved_checkpoints = train_info["saved_checkpoints"] 
            w_kl = train_info["kl_coef"]

            gpu_is_available = torch.cuda.is_available()
            if not gpu_is_available:
                print("[INFO] GPU not available, using CPU")
                server_config["device"] = "cpu"
            print("[INFO] Loading models from ", load_path, "- iter ", iter)
            checkpoint = torch.load(train_json_path+ f"iter_{iter}/checkpoint.pth",
                                    weights_only=True,
                                    map_location="cuda" if gpu_is_available else "cpu") 
     
        if server_config[ServerConstants.MODEL_ID] == 0:
            model_args = Launcher._get_mapzero_model_input_from_json(json_config)
            model = MapZero(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 1:
            model_args = Launcher._get_smartmap_model_input_from_json(json_config)
            model = SmartMap(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 2:
            model_args = Launcher._get_gprm_input_from_dict(json_config)
            model = GPRM(*model_args)
        elif server_config[ServerConstants.MODEL_ID] == 3:
            model_args = Launcher._get_transmap_input_from_dict(json_config)
            model = TransMap(*model_args)
        else:
            raise RuntimeError(f"Model not loaded - Model ID {server_config[ServerConstants.MODEL_ID]}")
        
        model = model.to(server_config["device"])
        model.load_state_dict(checkpoint["model_state"])

        supervised_loader_state = train_info.get(
            "supervised_loader_state", None
        )

        actor_lr = server_config["actor_lr"]
        actor_optimizer = Launcher.get_optimizer(server_config, model, actor_lr)
        actor_optimizer.load_state_dict(checkpoint["actor_optim_state"])

        critic_lr = server_config["critic_lr"]
        critic_optimizer = Launcher.get_optimizer(server_config, model, critic_lr)
        critic_optimizer.load_state_dict(checkpoint["critic_optim_state"])
            
        return model, actor_optimizer, critic_optimizer, cur_step, w_kl, trained_samples, last_saved_paths, num_saved_checkpoints, supervised_loader_state
    
    @staticmethod
    def set_seed(seed):
        if seed < 0:
            seed = 0
            print("[WARN] Seed is negative, setting to 0")
        print("Setting seed = ", seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)
        
    def load_dataset_from_checkpoint(self, server_config, json):
        if server_config.get("use_chunked_supervised_dataset", False):
            print("[INFO] Using chunked supervised dataset loader")
            if self.supervised_loader is None:
                self.supervised_loader = ChunkedSupervisedDatasetLoader(
                    server_config["chunked_dataset_path"],
                    device=server_config["device"]
                )
        else:
            if server_config["load_checkpoint_path"] == "":
                print(server_config)
                raise ValueError("Dataset path is empty")
            if not os.path.exists(server_config["load_checkpoint_path"]):
                raise ValueError(f"Dataset path {server_config["load_checkpoint_path"]} doesn't exist")
            print("[INFO] Loading dataset from ", server_config["load_checkpoint_path"])
            dataset = torch.load(server_config["load_checkpoint_path"] + "supervised_data.pth", map_location="cpu")
            self.dataset = dataset
            print("[INFO] Dataset loaded with total batches: ", len(self.dataset))

    def receive_supervised_data(self, server_config, json, tensors):
        if self.dataset is None:
            self.dataset = []

        self.dataset.append(tensors)
        if json["save_dataset"]:
            torch.save(self.dataset, server_config["save_checkpoint_path"] + "supervised_data.pth")
            
    def get_attention(self, server_config, json, tensors):
        attention_matrix = self.model.get_attention_and_output()
        
    
    
    def create_dataset_with_size(self, server_config, json):
        self.dataset = [None for _ in range(json["size"])]
    
    def receive_supervised_data_parallel(self, server_config, json, tensors):
        self.dataset[json["idx"]] = tensors
    
    def save_dataset(self, server_config, json):
        torch.save(self.dataset, json["path"] + "supervised_data.pth")
    
    def load_dataset(self, server_config, json):
        if server_config.get("use_chunked_supervised_dataset", False):
            if self.supervised_loader is None:
                print("[INFO] Using chunked supervised dataset loader")
                self.supervised_loader = ChunkedSupervisedDatasetLoader(
                    server_config["chunked_dataset_path"]                )
        else:
            self.dataset = torch.load(json["path"]+ "supervised_data.pth", map_location="cpu")

    def config_server(self, server_config:dict, json):
        self.w_kl = 1
        self.cur_step = 0
        self.trained_samples = 0

        server_config.update(json)
        print('*'*100)
        print("[INFO] Server config: ")
        print(server_config)
        print('*'*100)

        self.set_seed(server_config["seed"])
        if self.writer is not None:
            self.writer.close()
        self.writer = SummaryWriter(json[ServerConstants.TENSORBOARD_LOG_PATH])

        load_path = server_config["load_checkpoint_path"]
        if load_path != "":
            if not os.path.exists(load_path):
                raise ValueError("Checkpoint path doesn't exist")
            print("[INFO] Loading models from ", load_path)
            self.model, self.actor_optimizer,self.critic_optimizer, self.cur_step, self.w_kl, self.trained_samples, self.last_saved_paths, self.num_saved_checkpoints,supervised_loader_state = Launcher.load_model(server_config, load_path)
            if supervised_loader_state:
                print("[INFO] Restoring supervised dataset loader state")
                self.supervised_loader.load_state_dict(supervised_loader_state)
        else:
            self.model = Launcher.init_model(server_config)
            load_model_path = server_config["load_model_path"]
            if load_model_path != "":
                print("[INFO] Loading model only from ", load_model_path)
                self.load_nn_only(server_config, load_model_path)
            else:
                print("[INFO] No model loaded, initializing from scratch")
            if server_config["train_mode"]:
                self.actor_optimizer = Launcher.get_optimizer(server_config, self.model, server_config["actor_lr"])
                self.critic_optimizer = Launcher.get_optimizer(server_config, self.model,server_config["critic_lr"])
            
           
        if server_config["device"] == "cuda":
            self.model = self.model.to("cuda")

        config_text: str = server_config["config"] + "\n" + str(summary(self.model, verbose=0))
        del server_config["config"]
        self.writer.add_text("Experiment Configurations", config_text.replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;"))



    @staticmethod
    def get_train_data(server_config, tensors, is_mcts_train, is_ppo_train, is_transmap_train=False):
        data = {}
        final_idx = -1 
        
        if server_config["use_ISW"] and is_mcts_train:
            data["IS_weight"] = tensors[final_idx]
            final_idx -= 1 
        
        if  is_ppo_train or is_transmap_train:
            data["advantages"] = tensors[final_idx]
            data["actions"] = tensors[final_idx - 1]
            final_idx -= 2

        if not is_ppo_train and not is_mcts_train and not is_transmap_train:
            data["actions"] = tensors[final_idx]
            final_idx -= 1
        
        if is_mcts_train or is_ppo_train or is_transmap_train:
            data["policies"] = tensors[final_idx]
            final_idx -= 1

        if is_ppo_train:
            if not is_transmap_train:
                data["values"] = tensors[final_idx]
                final_idx -= 1
  
        data["target_values"] = tensors[final_idx]
        final_idx -= 1

        data["model_inputs"] = tensors[:final_idx + 1]

        if server_config["device"] == "cuda":
            for k,v in data.items():
                if k == "model_inputs":
                    data[k] = [tensor.to("cuda") for tensor in v]
                elif v is not None:
                    data[k] = v.to("cuda")
        else:
            assert server_config["device"] == "cpu"
        
        # for k,v in data.items():
        #     print(f"[DEBUG] {k}: {v.shape if isinstance(v, torch.Tensor) else 'list of tensors'}")
        #     if not isinstance(v,torch.Tensor):
        #         for t in v:
        #             print(f"    Tensor shape: {t.shape}")
        #     input()
       
        return data
    @staticmethod
    def ppo_train(values, old_values, target_values, old_probs, policy_logits, action_indices, advantages,
                  clip_factor):
        values = values.squeeze(-1)
        old_values = old_values.squeeze(-1)

        probs = torch.nn.functional.softmax(policy_logits, dim=-1)

        dist = Categorical(probs)
        old_dist = Categorical(old_probs)

        cur_log_prob = dist.log_prob(action_indices)
        log_old_probs = old_dist.log_prob(action_indices)

        ratio = torch.exp(cur_log_prob - log_old_probs)

        old_probs = old_probs.clamp(min=1e-8)
        new_probs = probs.clamp(min=1e-8)

        kl_loss = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(-1).mean()

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_factor, 1.0 + clip_factor)
        policy_loss = torch.max(policy_loss_1, policy_loss_2)

        entropy_loss = torch.mean(dist.entropy())

        value_clip = old_values + torch.clamp(values - old_values, -clip_factor, clip_factor)

        value_loss_1 = torch.nn.functional.mse_loss(target_values, values, reduction='none')
        value_loss_2 = torch.nn.functional.mse_loss(target_values, value_clip, reduction='none')
        value_loss = torch.max(value_loss_1, value_loss_2)

        return value_loss, policy_loss, entropy_loss, kl_loss   
    
    @staticmethod
    def ppo_train_transmap(values, target_values, old_probs, policy_logits, action_indices, advantages,
                  clip_factor):
        values = values.squeeze(-1)

        probs = torch.nn.functional.softmax(policy_logits, dim=-1)

        dist = Categorical(probs)
        old_dist = Categorical(old_probs)

        cur_log_prob = dist.log_prob(action_indices)
        log_old_probs = old_dist.log_prob(action_indices)

        ratio = torch.exp(cur_log_prob - log_old_probs)
        old_probs = old_probs.clamp(min=1e-8)

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_factor, 1.0 + clip_factor)
        policy_loss = torch.max(policy_loss_1, policy_loss_2)

        value_loss = torch.nn.functional.mse_loss(values, target_values, reduction='none')

        return value_loss, policy_loss,
    
    @staticmethod
    def supervised_train(predicted_values, target_values, predicted_logits, actions):

        predicted_values = predicted_values.squeeze(-1)
        target_values = target_values.squeeze(-1)

        value_loss = torch.nn.functional.mse_loss(
            predicted_values,
            target_values,
            reduction='none'
        )

        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        policy_loss = ce_loss_fn(predicted_logits, actions)

        predicted_actions = torch.argmax(predicted_logits, dim=-1)

        return value_loss, policy_loss, predicted_actions


    def get_supervised_data(self, server_config, json):
        if self.supervised_loader is not None:
            tensors = self.supervised_loader.next_batch()
        else:
            dataset_idx = json["cur_step"] % len(self.dataset)
            tensors = self.dataset[dataset_idx]

        return self.get_train_data(server_config, tensors, False, False)

    def train(self,server_config,json,tensors):
        # with sdpa_kernel(SDPBackend.MATH):
        train_id = server_config["train_id"] if json['train_id'] == -1 else json['train_id']
        
        if train_id == 3:
            if self.cur_step == 0 and server_config["lr_scheduler"] != "exponential":
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = 1e-9
            if self.dataset or self.supervised_loader:
                data = self.get_supervised_data(server_config, json)
            else:   
                data = self.get_train_data(server_config,tensors,False,False,False)
                
        else:
            data = self.get_train_data(server_config, tensors,train_id == 2,train_id == 0, train_id == 5)
        self.model.train()

        # self.critic_optimizer.zero_grad()
        if self.cur_step % self.grad_accumulate_steps == 0:
            self.actor_optimizer.zero_grad()

        policy_logits, values =  self.model(*data["model_inputs"])

        if server_config["buffer_id"] == 2:
            self.last_priorities = torch.abs(values - data["target_values"]) 

        if train_id == 2:
            value_loss, policy_loss = self.mcts_loss(values, data["target_values"], policy_logits, data["policies"])
        elif train_id == 1:
            value_loss, policy_loss = self.pg_train(values, data["target_values"], policy_logits, data["actions"], data["advantages"])
        elif train_id == 0:
            clip_factor = server_config[ServerConstants.CLIP_PARAM]
            value_loss, policy_loss, entropy_loss, kl_loss = self.ppo_train(values,data["values"], data["target_values"], 
                                                                            data["policies"], policy_logits, data["actions"],
                                                                            data["advantages"],clip_factor)
        elif train_id == 3:
            value_loss, policy_loss, predicted_actions = self.supervised_train(values, data["target_values"], policy_logits, data["actions"])   
            self.writer.add_scalar("Control/train_accuracy", (predicted_actions == data["actions"]).float().mean().item(), global_step=self.cur_step)
        elif train_id == 5:
            clip_factor = server_config[ServerConstants.CLIP_PARAM]
            value_loss, policy_loss = self.ppo_train_transmap(values, data["target_values"], 
                                                                            data["policies"], policy_logits, data["actions"],
                                                                            data["advantages"],clip_factor)
                                                                      
        else: 
            raise ValueError("Invalid train id")

        if "IS_weight" in data:
            isw = data["IS_weight"]
        else:
            isw = None

        if isw is not None:
            value_loss = isw*value_loss
            policy_loss = isw*policy_loss
        
        value_loss = value_loss.mean()
        policy_loss = policy_loss.mean()

        if not train_id == 0:
            total_loss = (value_loss + policy_loss) / self.grad_accumulate_steps
        else:
            w_v_loss = server_config[ServerConstants.VALUE_LOSS_COEF]
            w_entropy_loss = json[ServerConstants.ENTROPY_COEF]
            total_loss = (policy_loss - (entropy_loss * w_entropy_loss) + (value_loss * w_v_loss) + (kl_loss * self.w_kl)) / self.grad_accumulate_steps
            last_kl_coef = deepcopy(self.w_kl)
            self.w_kl = self.adjust_kl_penalty(kl_loss, self.w_kl, server_config[ServerConstants.KL_TARGET], 2)

        try:
            # with torch.autograd.set_detect_anomaly(True):
            total_loss.backward()
        except Exception as e:
            print(f"[ERROR] Error during backward pass: {e}")
            print("[ERROR] Policy Loss:", policy_loss.item())
            print("[ERROR] Policy: ", data["policies"])
            raise RuntimeError("Error during backward pass")



        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=server_config["max_grad_norm"],
            norm_type=2
        )

        grad_norm = torch.sqrt(sum(g.grad.data.norm(2)**2 for g in parameters))
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"[ERROR] Gradient for {name} contains NaN before step()")

        # is the global optimizer
        if (self.cur_step + 1) % self.grad_accumulate_steps == 0:

            self.actor_optimizer.step()
        # self.critic_optimizer.step()

        # for name, param in self.model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"[ERROR] Weight {name} became NaN after step()")

        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        
        actor_initial_lr = server_config["actor_lr"]
        # critic_initial_lr = server_config["critic_lr"]

        decay_rate = server_config[ServerConstants.LR_DECAY]
        decay_steps = server_config[ServerConstants.LR_DECAY_STEPS]
        actor_min_lr = server_config["actor_min_lr"]
        # critic_min_lr = server_config["critic_min_lr"]

        # cur_training_epoch = json["cur_step"]
        if server_config["lr_scheduler"] == "exponential":
            self.update_lr(self.actor_optimizer, actor_initial_lr, decay_rate, self.cur_step, decay_steps, actor_min_lr)
            # print(actor_initial_lr, actor_min_lr, self.cur_step)
            # print(actor_lr)
            # input("Other")
        else:
            self.update_lr_warmup_cosine(self.actor_optimizer, actor_initial_lr, actor_min_lr, self.cur_step + 1,
                                    server_config["warmup_steps"], server_config["total_steps"])
            # print(actor_initial_lr, actor_min_lr, self.cur_step)
            # print(actor_lr)
            # input("Supervised")
        # self.update_lr(self.critic_optimizer, critic_initial_lr, decay_rate, self.cur_step, decay_steps, critic_min_lr)


        ret =  {"Loss/total_loss":total_loss.item(), "Loss/policy_loss":policy_loss.item(), "Loss/value_loss":value_loss.item(),
                "Control/actor_lr": actor_lr,"Control/critic_lr": critic_lr, "Control/grad_norm": grad_norm.item(), "Control/raw_grad_norm":raw_grad_norm.item()}
        
        if train_id == 0:
            ret["Loss/entropy_loss"] = entropy_loss.item()
            ret["Loss/kl_loss"] = kl_loss.item()
            ret["Control/kl_coef"] = last_kl_coef
            ret["Control/entropy_coef"] = w_entropy_loss
        self.cur_step += 1
        self.trained_samples += policy_logits.shape[0]
        #torch.cuda.empty_cache()

        return ret
    
    # @staticmethod
    # def mcts_loss( pred_values, target_values, pred_logits, target_policies):


    #     value_loss = (target_values - pred_values.squeeze(-1))**2
                
    #     mask = (target_policies > 0).float()

    #     masked_logits = pred_logits * mask + (1 - mask) * (-1e9)
    #     # Check for NaN or Inf in masked_logits
    #     if torch.any(torch.isnan(masked_logits)) or torch.any(torch.isinf(masked_logits)):
    #         raise ValueError("NaN or Inf found in masked_logits")
    #     log_policy_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)

    #     policy_loss = (- log_policy_probs * target_policies).sum(-1)
    #     return value_loss, policy_loss
    @staticmethod
    def mcts_loss( pred_values, target_values, pred_logits, target_policies):
        # if pred_logits.shape[1] < target_policies.shape[1]:
        #     dim_size = target_policies.shape[1]
        #     diff = target_policies.shape[1] - pred_logits.shape[1]

        #     last_cols = target_policies[:, dim_size - diff:]
        #     assert (last_cols == -torch.inf).all(), \
        #         "target_policies must have -inf in the last dimensions"

        #     target_policies = target_policies[:, :pred_logits.shape[1]]

        # elif pred_logits.shape[1] > target_policies.shape[1]:
        #     raise ValueError("pred_logits cannot have more dimensions than target_policies")


        value_loss = (target_values - pred_values.squeeze(-1))**2
        target_policies = torch.where(target_policies == -torch.inf, 0, target_policies)
        masked_policy = torch.where(pred_logits == -torch.inf, -1e6,pred_logits)
        log_policy_probs = torch.nn.functional.log_softmax(masked_policy,dim=-1)
        policy_loss = (- log_policy_probs * target_policies).sum(-1)
        return value_loss, policy_loss

    @staticmethod
    def pg_train(pred_values, target_values, pred_logits, actions, advantages):

        value_loss = (target_values - pred_values.squeeze(-1))**2
        log_policy_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)

        policy_loss = (-log_policy_probs[torch.arange(0, pred_logits.size(0)), actions] * advantages)
        return value_loss, policy_loss

    def save_model(self, server_config, train_data):
        try:
            os.makedirs(server_config["save_checkpoint_path"], exist_ok=True)

            cur_iter = train_data["cur_iter"]

            last_checkpoint_idx = (self.num_saved_checkpoints) % self.max_checkpoints_to_save
            prev_path = self.last_saved_paths[last_checkpoint_idx]
            checkpoint_path = os.path.join(server_config["save_checkpoint_path"], f"iter_{cur_iter}")
            self.last_saved_paths[last_checkpoint_idx] = checkpoint_path
            self.num_saved_checkpoints += 1 
            train_info = {
                "final_iter": cur_iter,
                "train_finished": train_data["train_finished"],
                "cur_step": self.cur_step,
                "trained_samples": self.trained_samples,
                "entropy_coef": train_data["entropy_coef"],
                "kl_coef": self.w_kl,
                "ema_rout_nodes": train_data["ema_rout_nodes"],
                "ema_valid_map_rate": train_data["ema_valid_map_rate"],
                "last_saved_paths": self.last_saved_paths.copy(), 
                "saved_checkpoints": self.num_saved_checkpoints       
            }
            os.makedirs(checkpoint_path, exist_ok=True)

            if prev_path and os.path.isdir(prev_path) and prev_path.startswith(server_config["save_checkpoint_path"]):
                try:
                    shutil.rmtree(prev_path)
                except Exception as e:
                    print(f"[WARN] Error {prev_path}: {e}")


            torch.save({
                "model_state": self.model.state_dict(),
                "actor_optim_state": self.actor_optimizer.state_dict(),
                "critic_optim_state": self.critic_optimizer.state_dict()
            }, os.path.join(checkpoint_path, "checkpoint.pth"))

            if not os.path.exists(os.path.join(server_config["save_checkpoint_path"], "config.json")):
                Launcher.write_json(server_config["save_checkpoint_path"], server_config, "config")
            if self.supervised_loader is not None:
                train_info["supervised_loader_state"] = \
                    self.supervised_loader.state_dict()
            Launcher.write_json(server_config["save_checkpoint_path"], train_info, "train_info")

           

        except Exception as e:
            print(f"[Error] {e}")
            raise RuntimeError("Error saving the model")


    @staticmethod
    def update_lr(optimizer, initial_lr, decay_rate, cur_training_step, decay_steps, min_lr):
        lr = initial_lr * decay_rate ** (
                cur_training_step  / decay_steps
            )
        # print("[DEBUG] Updated LR before min check: ", lr)
        # print("[DEBUG] initial_lr:", initial_lr, " decay_rate:", decay_rate, " cur_training_step:", cur_training_step, " decay_steps:", decay_steps)
        lr = max(lr, min_lr)
        # print("[DEBUG] Computed LR: ", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @staticmethod
    def update_lr_warmup_cosine(optimizer, max_lr, min_lr,
                            cur_training_step, warmup_steps, total_steps):
        import math
        
        if cur_training_step < warmup_steps:
            lr = max_lr * (cur_training_step / warmup_steps)
        else:
            progress = (cur_training_step - warmup_steps) / (total_steps - warmup_steps)
            
            progress = min(max(progress, 0.0), 1.0)
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (max_lr - min_lr) * cosine_decay

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # print("[DEBUG] Computed LR (warmup cosine): ", lr)
        # print("[DEBUG] max_lr:", max_lr, " min_lr:", min_lr, " cur_training_step:", cur_training_step, " warmup_steps:", warmup_steps, " total_steps:", total_steps)
        return lr
            
            
    def adjust_kl_penalty(self, kl_divergence, w_kl, target_kl,kl_factor):
        if kl_divergence > target_kl * 1.5:
            w_kl *= kl_factor 
        elif kl_divergence < target_kl / 1.5 and w_kl > 0.01:
            w_kl /= kl_factor
        w_kl = min(max(w_kl, 0.01), 10)
        return w_kl
    
    def write_tensorboard(self, server_config, json):

        def plot_discrete_histogram_from_counts(counts, title):
            fig_width = 12
            dpi = 150
            plt.figure(figsize=(fig_width, 6))

            x_values = np.arange(len(counts))
            y_values = counts

            plt.bar(x_values, y_values, edgecolor='black', width=1.0)

            fig_pixel_width = fig_width * dpi
            max_label_width_px = 20  
            max_ticks = max(1, fig_pixel_width // max_label_width_px)
            step = max(1, len(x_values) // max_ticks)
            tick_positions = x_values[::step]

            plt.xticks(tick_positions, rotation=45, ha='right')
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, axis='y')

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=dpi)
            plt.close()

            buf.seek(0)
            image = Image.open(buf).convert("RGB").copy()
            buf.close()

            image_np = np.array(image).transpose(2, 0, 1)
            return image_np



        def plot_discrete_histogram_image(values, title):
            fig_width = 8
            dpi = 150
            plt.figure(figsize=(fig_width, 5))
            MAX_BINS = 500

            min_val = int(min(values))
            max_val = int(max(values))
            range_size = max_val - min_val + 1

            if range_size > MAX_BINS:
                bins = MAX_BINS
            else:
                bins = np.arange(min_val, max_val + 2) - 0.5
            # bins = np.arange(min_val, max_val + 2) - 0.5

            plt.hist(values, bins=bins, edgecolor='black', rwidth=0.9)

            full_range = list(range(min_val, max_val + 1))
            fig_pixel_width = fig_width * dpi
            max_label_width_px = 20
            max_ticks = max(1, fig_pixel_width // max_label_width_px)
            step = max(1, len(full_range) // max_ticks)
            tick_positions = full_range[::step]

            plt.xticks(tick_positions, rotation=45, ha='right')
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, axis='y')

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=dpi)
            plt.close()

            buf.seek(0)
            image = Image.open(buf).convert("RGB").copy()
            buf.close()

            image_np = np.array(image).transpose(2, 0, 1)
            return image_np
        step = json["step"]

        for k,v in json["Train"].items():
            self.writer.add_scalar(k + " x step", v, step) 

        for k,v in json["Eval"].items():
            self.writer.add_scalar(k + " x step", v, step) 

        for k, v in json["TrainHist"].items():
            if "Control/sampled_idx_count" in k:
                image = plot_discrete_histogram_from_counts(v, f"{k} - Train (Step {step})")
                self.writer.add_image(f"{k}_Control_Histogram", image, step)
            elif "Control/dims_to_action_count_" in k:
                image = plot_discrete_histogram_from_counts(v, f"{k} - Train (Step {step})")
                self.writer.add_image(f"{k}_Control_Histogram", image, step)
            else:
                image = plot_discrete_histogram_image(v, f"{k} - Train (Step {step})")
                self.writer.add_image(f"{k}_Train_Histogram", image, step)

        for k, v in json["TestHist"].items():
            image = plot_discrete_histogram_image(v, f"{k} - Test (Step {step})")
            self.writer.add_image(f"{k}_Test_Histogram", image, step)
            
        self.writer.add_scalar("Control/train_steps", self.cur_step, step)
        self.writer.add_scalar("Control/trained_samples", self.trained_samples, step)

        return None
    
    def finish_server(self,server_config, none_return):
        self.supervised_loader.stop()
        return None
    
    @staticmethod
    def get_optimizer(server_config, model: torch.nn.Module, lr):
        optimizer = server_config["optimizer"]
        weight_decay = server_config["weight_decay"]

        if optimizer == 0:
            return torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay)

        if optimizer == 1:
            return torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
        raise ValueError("Invalid optimizer")
    
    @staticmethod
    def _get_mapzero_model_input_from_json(json_data: dict):
        dim_feat_dfg = json_data["in_dim_feat_dfg"]
        dim_feat_cgra = json_data["in_dim_feat_cgra"]
        out_dim = json_data["out_dim"]
        n_heads = json_data["n_heads"]
        number_of_spatial_PEs = json_data["number_of_spatial_PEs"]
        negative_slope = json_data["negative_slope"]
        return [dim_feat_dfg, dim_feat_cgra, out_dim, n_heads, number_of_spatial_PEs, negative_slope]
    
    @staticmethod
    def _get_smartmap_model_input_from_json(json_data: dict):
        dim_feat_dfg = json_data["in_dim_feat_dfg"]
        dim_feat_cgra = json_data["in_dim_feat_cgra"]
        out_dim = json_data["out_dim"]
        n_heads = json_data["n_heads"]
        negative_slope = json_data["negative_slope"]
        return [dim_feat_dfg, dim_feat_cgra, out_dim, n_heads, negative_slope]
    
    @staticmethod
    def _get_gprm_input_from_dict(server_config: dict):
        lpe_dim = server_config["lpe_dim"]
        vocab_len = server_config["vocab_len"]
        feat_out_dim = server_config["feat_out_dim"]
        out_dim = server_config["out_dim"]
        ff_out_dim = server_config["ff_out_dim"]
        GIN_layers_list = server_config["GIN_layers_list"]
        n_graph_embed_layers = server_config["n_graph_embed_layers"]
        n_transformer_layers = server_config["n_transformer_layers"]
        n_heads = server_config["n_heads"]
        activation_fn = server_config["activation_func"]
        use_mlp = server_config["use_mlp"]
        dropout = server_config["dropout"]
        use_rms_norm = server_config["use_rms_norm"]
        use_norm_before_pred = server_config["use_norm_before_pred"]
        negative_slope = server_config["negative_slope"]
        max_len = server_config["max_len"]
        norm_initial_emb = server_config["norm_initial_emb"]
        use_gin_emb = server_config["use_gin_emb"]
        use_mobility_info = server_config["use_mobility_info"]
        use_coord_info = server_config["use_coord_info"]
        use_gnn = server_config["use_gnn"]
        use_transformer = server_config["use_transformer"]
        use_placement_order = server_config["use_placement_order"]
        ignore_lpe = server_config["ignore_lpe"]
        scale = server_config["use_scale"]

        return [ lpe_dim, vocab_len, feat_out_dim,
                 out_dim, ff_out_dim, GIN_layers_list, n_graph_embed_layers, n_transformer_layers, n_heads, activation_fn , use_mlp,
                 dropout , use_rms_norm , use_norm_before_pred, negative_slope, max_len,
                 norm_initial_emb , use_gin_emb , 
                  use_mobility_info, use_coord_info, use_gnn,
                 use_transformer, use_placement_order, ignore_lpe, scale]
    
    @staticmethod
    def _get_transmap_input_from_dict(server_config: dict):
        max_dfg_feat = server_config["max_dfg_feat"]
        max_adg_feat = server_config["max_adg_feat"]
        lpe_dim = server_config["lpe_dim"]
        E_dfg_node_dim = server_config["E_dfg_node_dim"]
        E_dfg_edge_dim = server_config["E_dfg_edge_dim"]
        E_adg_node_dim = server_config["E_adg_node_dim"]
        E_adg_edge_dim = server_config["E_adg_edge_dim"]
        out_dim_policy = server_config["out_dim_policy"]
        state_dim = server_config["state_dim"]
        out_dim = server_config["out_dim"]
        ff_dim = server_config["ff_dim"]
        n_heads = server_config["n_heads"]
        nb_features = server_config["nb_features"]
        hidden_dim = server_config["hidden_dim"]
        ortho_scaling = server_config["ortho_scaling"]
        dropout = server_config["dropout"]
        activation_fn = server_config["activation_fn"]
        num_mhd_layers = server_config["num_mhd_layers"]

        return [num_mhd_layers,
            max_dfg_feat, max_adg_feat, lpe_dim,
            E_dfg_node_dim, E_dfg_edge_dim, E_adg_node_dim, E_adg_edge_dim,
            out_dim_policy, state_dim, out_dim, ff_dim,
            n_heads, nb_features, hidden_dim,
            ortho_scaling, dropout, activation_fn
        ]


    @staticmethod
    def write_json(path_to_write: str, json_data, name):
        with open(path_to_write + f"{name}.json", "w") as f:
            json.dump(json_data,f)
        
    @staticmethod
    def load_config(path_to_load):
        if "iter" in path_to_load:
            path_to_load = os.path.dirname(path_to_load) + "/"
        with open(path_to_load + "config.json", "r") as f:
            json_params = json.load(f)
        return json_params
        
       
      
    @staticmethod
    def gen_laplacian_embeddings(server_config, json, tensors):
        receiver_return = [tensors[0].long(),
            json["num_nodes"],
            json["is_undirected"],
            json["D_dim"]
            ]
        evals, evects = get_laplacian_emb(*receiver_return) 
        # print(evects)
        # input()
        return tensor_to_msgpack(evects)
