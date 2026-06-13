import torch
import torch.nn as nn
import math

import torch
from torch import nn
import numpy as np

def get_positional_encoding(maxlen: int, emb_size: int) -> torch.Tensor:
    den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
    pos = torch.arange(0, maxlen).unsqueeze(1)
    pe = torch.zeros(maxlen, emb_size)
    pe[:, 0::2] = torch.sin(pos * den)
    pe[:, 1::2] = torch.cos(pos * den)
    return pe

class PlacementOrderEncoder(nn.Module):
    def __init__(self, pos_embedding: torch.Tensor, out_dim: int):
        super().__init__()
        self.register_buffer("pos_embedding", pos_embedding)
        in_dim = pos_embedding.size(1)

        self.proj = nn.Linear(in_dim, out_dim)
        self.resid_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, placement_order_ids: torch.LongTensor):
        x = self.pos_embedding[placement_order_ids]      # (N, in_dim)
        projected = self.proj(x)                        # (N, out_dim)
        residual = self.resid_proj(x)                   # (N, out_dim)
        out = self.ln(projected + residual)             # skip + layernorm
        return out

class MobilityEncoder(nn.Module):
    def __init__(self, pos_embedding: torch.Tensor, out_dim: int):
        super().__init__()
        self.register_buffer("pos_embedding", pos_embedding)
        in_dim = pos_embedding.size(1)

        self.asap_proj = nn.Linear(in_dim, out_dim)
        self.alap_proj = nn.Linear(in_dim, out_dim)

        self.resid_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.ln = nn.LayerNorm(out_dim)

    def forward(self, temporal_ids: torch.LongTensor):
        asap_ids = temporal_ids[0]
        alap_ids = temporal_ids[1]

        asap = self.pos_embedding[asap_ids]     # (N, in_dim)
        alap = self.pos_embedding[alap_ids]     # (N, in_dim)

        asap_emb = self.asap_proj(asap)         # (N, out_dim)
        alap_emb = self.alap_proj(alap)         # (N, out_dim)

        projected = asap_emb + alap_emb         
        residual = self.resid_proj(asap + alap) 

        out = self.ln(projected + residual)
        return out


class PEPositionalEncoder(nn.Module):
    def __init__(self, pos_embedding: torch.Tensor, out_dim: int):
        super().__init__()

        self.register_buffer("pos_embedding", pos_embedding)
        in_dim = pos_embedding.size(1)
        
        self.proj_x = nn.Linear(in_dim, out_dim)
        self.proj_y = nn.Linear(in_dim, out_dim)
        self.proj_z = nn.Linear(in_dim, out_dim)

        self.resid_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.ln = nn.LayerNorm(out_dim) 

    def forward(self, pos_x, pos_y, pos_z):
        ex = self.pos_embedding[pos_x]      
        ey = self.pos_embedding[pos_y]
        ez = self.pos_embedding[pos_z]

        projected = (
            self.proj_x(ex) +
            self.proj_y(ey) +
            self.proj_z(ez)
        )

        residual = self.resid_proj(ex + ey + ez)

        out = self.ln(projected + residual)
        return out
    
class LPEEncoder(nn.Module):
    def __init__(self, lpe_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(lpe_dim, out_dim)
        self.resid_proj = nn.Linear(lpe_dim, out_dim) if lpe_dim != out_dim else nn.Identity()
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, lpe):
        projected = self.proj(lpe)
        residual = self.resid_proj(lpe)
        out = self.ln(projected + residual)
        return out