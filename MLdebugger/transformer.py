import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # embedding
        self.conv1 = nn.Linear(input_size, embed_dim)
        self.conv2 = nn.Linear(embed_dim, embed_dim)

        # positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.tranformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # padding
        # x = F.pad(x, (0, self.embed_dim - self.input_size))
        x = self.conv1(x)
        x = F.relu(x)
        c2 = self.conv2(x)
        x = F.relu(x + c2)

        # scaling input 
        x = x * (self.embed_dim ** 0.5)

        # positional encoding
        x = self.pos_encoder(x)

        # transformer encoder
        x = self.tranformer_encoder(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()*(-torch.log(torch.tensor(10000.0))/embed_dim))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
