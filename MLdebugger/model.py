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

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
        

class CombinedModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1, mode="dot"):
        super(CombinedModel, self).__init__()
        self.graphsage = GraphSAGE(in_channels, hidden_channels)
        self.bilstm = BiLSTM(in_channels, hidden_channels//2, num_layers)
        self.mode = mode

    def forward(self, trace, x, edge_index):
        embeddings = self.graphsage(x, edge_index)
        out_bilstm = self.bilstm(trace)
        embeddings_expanded = embeddings.unsqueeze(0).expand(out_bilstm.size(0), -1, -1)
        
        # dot product
        if self.mode == "dot":
            combined = torch.bmm(out_bilstm, embeddings_expanded.transpose(1, 2)) # B * T * N
        # cosine similarity
        elif self.mode == "cos":
            out_bilstm = F.normalize(out_bilstm, p=2, dim=-1) # B * T * E
            embeddings_expanded = F.normalize(embeddings_expanded, p=2, dim=-1) # B * N * E
            combined = torch.bmm(out_bilstm, embeddings_expanded.transpose(1, 2)) # B * T * N
        # -euclidean distance
        else:
            #print(out_bilstm.unsqueeze(2).size(), embeddings_expanded.unsqueeze(1).size())
            combined = -torch.norm(out_bilstm.unsqueeze(2) - embeddings_expanded.unsqueeze(1), dim=-1)
            
        
        return combined
