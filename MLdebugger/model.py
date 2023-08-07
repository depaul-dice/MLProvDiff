import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch

class BiLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
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

        # nn skip connections
        self.skip1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels // 2), 
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )
        self.skip2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )

        # BN layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1 + self.skip1(x))
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.skip2(x1))
        x2 = self.dropout(x2)
        return x2

class CombinedModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super(CombinedModel, self).__init__()
        self.graphsage = GraphSAGE(in_channels, hidden_channels)
        self.bilstm = BiLSTM(in_channels, hidden_channels//2, num_layers)

    def forward(self, trace, x, edge_index):
        embeddings = self.graphsage(x, edge_index)
        out_bilstm = self.bilstm(trace)
        embeddings_expanded = embeddings.unsqueeze(0).expand(out_bilstm.size(0), -1, -1)
        combined = torch.bmm(out_bilstm, embeddings_expanded.transpose(1, 2))
        return combined
