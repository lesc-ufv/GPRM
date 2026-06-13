from torch import nn

class GPRMMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, activation_fn=nn.ReLU()):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, 2*in_dim)
        self.activation1 = activation_fn
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(2*in_dim, in_dim)
        self.activation2 = activation_fn
        self.dropout2 = nn.Dropout(dropout)

        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x

        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        x = x + residual 

        return self.out(x)
