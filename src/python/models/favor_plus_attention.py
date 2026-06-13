import math
import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial

#  adapted from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, _ = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    return q.t().to(device)

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = nb_rows // nb_columns
    block_list = [orthogonal_matrix_chunk(nb_columns, device=device) for _ in range(nb_full_blocks)]

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix



def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)


def linear_attention(q, k, v, mask=None):
    if exists(mask):
        mask = mask[:, None, :, None].float()
        q = q * mask
        k = k * mask
        v = v * mask

    kv = torch.einsum("...nd,...ne->...de", k, v)

    k_sum = k.sum(dim=-2) 
    z = 1.0 / (torch.einsum("...nd,...d->...n", q, k_sum) + 1e-6)

    out = torch.einsum("...de,...nd,...n->...ne", kv, q, z)

    return out



class FAVORPlusAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        nb_features=None,
        ortho_scaling=0,
        dropout=0.0,
        qkv_bias=True,
        attn_out_bias=True,
    ):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dim_head = dim // heads

        nb_features = nb_features or int(self.dim_head * math.log(self.dim_head))
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=nb_features,
            nb_columns=self.dim_head,
            scaling=ortho_scaling,
        )

        self.register_buffer("projection_matrix", self.create_projection())

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=attn_out_bias)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _ = x.size()

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        create_kernel = partial(
            softmax_kernel,
            projection_matrix=self.projection_matrix,
            device=x.device
        )

        q = create_kernel(q, is_query=True)
        k = create_kernel(k, is_query=False)

        out = linear_attention(q, k, v, mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)

        return self.dropout(self.norm(x + out))

