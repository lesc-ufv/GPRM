import torch

class ValueNet(torch.nn.Module):
    def __init__(self,dim_input,hiddend_dim,leaky_relu_slope):
        super(ValueNet,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_input,hiddend_dim),
            torch.nn.Linear(hiddend_dim,hiddend_dim)
            )
        self.fc = torch.nn.Linear(hiddend_dim,1)
        self.slope_leaky_relu = leaky_relu_slope

    def forward(self,state_vector):
        x = self.mlp(state_vector)
        x = torch.nn.functional.leaky_relu(x,self.slope_leaky_relu)
        x = self.fc(x)
        return x
    
    def print(self):
        print("Last layer weights:",  self.fc.weight, self.fc.weight.mean().item(), self.fc.weight.std().item())
        for name, param in self.named_parameters():
            if param.grad is not None:
              print (param.grad)
        input()
