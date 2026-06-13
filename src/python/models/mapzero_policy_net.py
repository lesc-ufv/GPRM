import torch
import torch.nn as nn

class MapZeroPolicyNet(nn.Module):
    def __init__(self, dim_input, dim_out, action_space_size, slope_leaky_relu=0.01):
        super(MapZeroPolicyNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_out),
            nn.LeakyReLU(negative_slope=slope_leaky_relu),
            nn.Linear(dim_out, dim_out),
            nn.LeakyReLU(negative_slope=slope_leaky_relu)
        )
        self.fc = nn.Linear(dim_out, action_space_size)
        self.slope_leaky_relu = slope_leaky_relu
        

    def forward(self, state_vector):
        x = self.mlp(state_vector)
        x = self.fc(x)

        return x  