import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
from transformer import TransformerEncoderModel

class BiLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
        
class CombinedModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, encoder, num_heads, dropout):
        super(CombinedModel, self).__init__()
        self.graphsage = GraphSAGE(in_channels, hidden_channels)
        
        if encoder == 'lstm':
            self.encoder = BiLSTM(in_channels, hidden_channels//2, num_layers, dropout)
        elif encoder == 'transformer':
            self.encoder = TransformerEncoderModel(in_channels, hidden_channels, num_layers, num_heads, dropout)
        else:
            raise Exception(f'Encoder named "{encoder}" not supported. Please choose from "lstm" or "transformer"')

    def forward(self, trace, x, edge_index):
        embeddings = self.graphsage(x, edge_index)
        out_encoder = self.encoder(trace)
        embeddings_expanded = embeddings.unsqueeze(0).expand(out_encoder.size(0), -1, -1)
        combined = torch.bmm(out_encoder, embeddings_expanded.transpose(1, 2)) # B * T * N
        
        return combined
