#ifndef SERVER_KEYS
#define SERVER_KEYS

#include <iostream>
#include <string>
class ServerConstants{
public:
   std::string args_k ="args";
   std::string debug_k ="debug_mode";
   std::string func_k = "func";
   std::string tensors_k ="tensors";
   std::string model_id_k ="model_id";
   std::string dtype_k ="dtype";
   std::string shape_k ="shape";
   std::string data_k ="data_k";
   std::string device_k = "device";
   std::string save_checkpoint_name_k = "save_checkpoint_name";
   std::string load_checkpoint_path_k = "load_checkpoint_path";
   std::string save_checkpoint_path_k = "save_checkpoint_path";
   std::string tensorboard_log_path_k = "tensorboard_log_path";

   std::string NONE = "None";
   std::string SEED = "seed";

   std::string clip_param_k = "clip_param";
   std::string value_loss_coef_k = "value_loss_coef";
   std::string entropy_coef_k = "entropy_coef";
   std::string max_grad_norm_k = "max_grad_norm";
   std::string optimizer_epsilon_k = "optimizer_epsilon";
   std::string lr_decay_k = "lr_decay";
   std::string lr_decay_steps_k = "lr_decay_steps";
   std::string min_lr_k = "min_lr";
   std::string kl_target_k = "kl_target";
   std::string kl_coefficient_k = "kl_coefficient";
   std::string epochs_k = "epochs";
   std::string batch_size_k = "batch_size";
   std::string learning_rate_k = "learning_rate";
   std::string out_dim_k = "out_dim";
   std::string n_heads_k = "n_heads";
   std::string in_dim_feat_dfg_k = "in_dim_feat_dfg";
   std::string in_dim_feat_cgra_k = "in_dim_feat_cgra";
   std::string negative_slope_k = "negative_slope";
   std::string len_action_space_k = "action_space";


};
#endif  
