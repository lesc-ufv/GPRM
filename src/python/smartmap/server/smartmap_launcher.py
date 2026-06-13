from copy import deepcopy
from torch.distributions import Categorical
import torch
from src.python.constants.server_constants import ServerConstants
import os
from torch.utils.tensorboard import SummaryWriter

class SmartMapLauncher:
    def __init__(self):
        self.old_model = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.writer = None 
        self.w_kl = 1

    
    def config_server(self,server_config, json):
        print(server_config)
        seed = server_config[ServerConstants.SEED]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) 
        self.writer = SummaryWriter(server_config[ServerConstants.TENSORBOARD_LOG_PATH])
        return None
    
    def initialize_models(self,server_config,receiver_return):
        self.model,self.optimizer = receiver_return
        self.model = self.model.to(server_config[ServerConstants.DEVICE])
        self.old_model = deepcopy(self.model)
    
    def smartmap_infer(self, server_config, tensors):
        self.model.eval()
        if server_config[ServerConstants.DEVICE] == "cuda":
            tensors = [tensor.to(server_config[ServerConstants.DEVICE]) for tensor in tensors]
        with torch.no_grad():
            return self.model(*tensors)
        
    def update_old_model(self,server_config, receier_return):
        self.old_model = deepcopy(self.model)

    def ppo_training(self, server_config, receiver_return):
        self.model.train()
        self.optimizer.zero_grad()
        
        w_v_loss = server_config[ServerConstants.VALUE_LOSS_COEF]
        clip_factor = server_config[ServerConstants.CLIP_PARAM]
        w_entropy_loss = receiver_return[ServerConstants.ENTROPY_COEF]
        debug_mode = server_config[ServerConstants.DEBUG_MODE]

        inputs = receiver_return["model_inputs"]
        target_values = receiver_return["target_values"]
        action_indices = receiver_return["actions_indices"]
        advantages = receiver_return["advantages"]
        if debug_mode:
            print("Inputs:", inputs)
            print("Target values:", target_values)
            print("Action indices:", action_indices)
            print("Device:", server_config[ServerConstants.DEVICE])

        if server_config[ServerConstants.DEVICE] == "cuda":
            inputs = [tensor.to(server_config[ServerConstants.DEVICE]) for tensor in inputs]
            action_indices = action_indices.to(server_config[ServerConstants.DEVICE])
            target_values = target_values.to(server_config[ServerConstants.DEVICE])
            advantages = advantages.to(server_config[ServerConstants.DEVICE])
        with torch.no_grad():
            old_policy_logits, old_values = self.old_model(*inputs)
        
        policy_logits, values = self.model(*inputs)
        values = values.squeeze(-1)
        old_values = old_values.squeeze(-1)
        old_probs = torch.nn.functional.softmax(old_policy_logits,dim=-1)
        probs = torch.nn.functional.softmax(policy_logits,dim=-1) 
        probs = torch.clamp(probs, min=1e-10, max=1.0) 
        probs = probs / probs.sum(dim=-1, keepdim=True)  
        old_probs = torch.clamp(old_probs, min=1e-10, max=1.0) 
        old_probs = old_probs / old_probs.sum(dim=-1, keepdim=True)  

        if advantages.size(0) > 1:
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        
        if debug_mode:
            print("Old policy logits:", old_policy_logits)
            print("Policy_logits:", policy_logits)
            print("Old probs:", old_probs)
            print("Probs:", probs)
            print("Old values:", old_values)
            print("Values:", values)
            print("Advantages:", advantages)

        dist = Categorical(probs)
        old_dist = Categorical(old_probs)

        cur_log_prob = dist.log_prob(action_indices)
        log_old_probs = old_dist.log_prob(action_indices)
        if debug_mode:
            print("Cur log prob:", cur_log_prob)
            print("Log old probs:", log_old_probs)

        ratio = torch.exp(cur_log_prob - log_old_probs)
        kl_loss = torch.distributions.kl_divergence(old_dist, dist).mean()
        # if torch.isinf(kl_loss):
        #     print("probs:\n",probs)
        #     print("old_probs:\n",probs)
        #     print("div:\n")

        if debug_mode:
            print("Ratio:", ratio)
            print("KL loss:", kl_loss)

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_factor, 1.0 + clip_factor)
        policy_loss = torch.mean(torch.max(policy_loss_1, policy_loss_2))


        entropy_loss = torch.mean(dist.entropy())
        value_clip = old_values + torch.clamp(values - old_values, -clip_factor, clip_factor)


        value_loss_1 = torch.nn.functional.mse_loss(target_values, values,reduction = 'none')
        value_loss_2 = torch.nn.functional.mse_loss(target_values, value_clip, reduction = 'none')
        value_loss = torch.mean(torch.max(value_loss_1, value_loss_2))

        total_loss = policy_loss - (entropy_loss * w_entropy_loss) + (value_loss * w_v_loss) + (kl_loss * self.w_kl)

        total_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=server_config[ServerConstants.MAX_GRAD_NORM], norm_type=2
        )


        self.optimizer.step()

        last_kl_coef = deepcopy(self.w_kl)
        self.w_kl = self.adjust_kl_penalty(kl_loss, self.w_kl, server_config[ServerConstants.KL_TARGET], 2)

        initial_lr = server_config[ServerConstants.LEARNING_RATE]
        decay_rate = server_config[ServerConstants.LR_DECAY]
        decay_steps = server_config[ServerConstants.LR_DECAY_STEPS]
        min_lr = server_config[ServerConstants.MIN_LR]
        cur_training_epoch = receiver_return["cur_epoch"]

        lr = self.update_lr(self.optimizer, initial_lr, decay_rate, cur_training_epoch, decay_steps, min_lr)

        total_norm = torch.norm(
            torch.stack(
                [
                    param.grad.norm(2)
                    for param in self.model.parameters()
                    if param.grad is not None
                ]
            )
        ).item()
        if debug_mode:
            print("Policy loss:", policy_loss)
            print("Entropy loss:", -entropy_loss)
            print("Value clip:", value_clip)
            print("Value loss 1:", value_loss_1)
            print("Value loss 2:", value_loss_2)
            print("Value loss:", value_loss)
            print("Total loss:", total_loss)
            print("Gradient norm:", norm)
            print("Learning rate:", lr)
            print("KL weigth: ", last_kl_coef)
            print("Entropy weigth: ", w_entropy_loss)
            print("Total gradient norm:", total_norm)
        return total_loss.item(), value_loss.item(), policy_loss.item(), entropy_loss.item(), kl_loss.item(), last_kl_coef, total_norm, lr, w_entropy_loss


    @staticmethod
    def update_lr(optimizer, initial_lr, decay_rate, cur_training_step, decay_steps, min_lr):
        lr = initial_lr * decay_rate ** (
                cur_training_step  / decay_steps
            )
        lr = max(lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def adjust_kl_penalty(self, kl_divergence, w_kl, target_kl,kl_factor):
        if kl_divergence > target_kl * 1.5:
            w_kl *= kl_factor 
        elif kl_divergence < target_kl / 1.5:
            w_kl /= kl_factor
        w_kl = max(min(w_kl, 10),target_kl)
        return w_kl

    def write_tensorboard(self, server_config, receiver_return):
        step = receiver_return["step"]
        for k,v in receiver_return["Train"].items():
            self.writer.add_scalar("Train/" + k + " x step", v, step) 

        for k,v in receiver_return["Eval"].items():
            self.writer.add_scalar("Eval/" + k + " x step", v, step) 
        return None
    
    def save_model(self, server_config, none_return):
        if not os.path.exists(server_config[ServerConstants.CHECKPOINT_PATH]):
            os.makedirs(server_config[ServerConstants.CHECKPOINT_PATH])
        return torch.save(self.model,server_config[ServerConstants.CHECKPOINT_PATH] + server_config[ServerConstants.CHECKPOINT_NAME] + ".pth")

    def finish_server(self,server_config, none_return):
        return None