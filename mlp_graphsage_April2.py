import random
import numpy as np

from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import SGConv
import pickle

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os

device = f'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


class GraphSAGE(torch.nn.Module):

    """
    input dimension: dimension of the feature vector
    output dimension: dimension of the node (this should be equal to the dmension of the trace)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, data):
        x, adj_t = data.x, data.edge_index
        x = self.conv1(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        #return torch.log_softmax(x, dim=-1)
        return x

class MLP(torch.nn.Module):
    """
    This will the one-hot encoded labels of all the nodes in the graph and output 
    a output_dim long vector

    input dim: vocab size
    output dim: 100 (a hyperparameter)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 200)
        self.hidden_fc = nn.Linear(200,150)
        self.output_fc = nn.Linear(150, output_dim)

    def forward(self, x):
        # x = [num_nodes,vocab_size] (412, 412)
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [num_nodes, 200] (412, 200)

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [num_nodes, 150] (412, 150)

        output_mlp = self.output_fc(h_2)
        #output_mlp = [num_nodes, output_dim] (412, 100)

        return output_mlp

def train(model_mlp, optimizer_mlp, 
        model_graphsage, optimizer_graphsage,
        data_nodeLabels, data_adjacencyMatrix):

    # call the mlp class
    model_mlp.train()
    optimizer_mlp.zero_grad()
    out_mlp = model_mlp(data_nodeLabels)

    # create a graphsage custom data
    data = Data(x=out_mlp, edge_index=adj_2d.t().contiguous())

    #call graphsage
    model_graphsage.train()
    optimizer_graphsage.zero_grad()
    out_graphsage = model_graphsage(data)
    print(out_graphsage.shape)

    return "Succesful"

def structure_adjacency_matrix():
    """
    This method is to convert the original adjacency matrix into a 2 dimensional 
    adjacency matrix
    """
    adj_2d = []
    t_ind = 0
    for t in data_adjacencyMatrix:
        elem_ind = 0
        for elem in t:
            if elem == 1:
                a= [t_ind, elem_ind]
                adj_2d.append(a)
            elem_ind += 1
        t_ind = t_ind+1
    return torch.tensor(adj_2d)


data_nodeLabels = pickle.load(open("/home/dhruvs/depaul_data/nodeLabels.pkl", "rb"))
data_adjacencyMatrix_old = pickle.load(open("/home/dhruvs/depaul_data/adjacencyMatrix.pkl", "rb"))
data_adjacencyMatrix = data_adjacencyMatrix_old.type(torch.int64)


lr = 1e-4 
epochs = 1

in_dim_mlp = data_nodeLabels.shape[1] #current vocab size
output_dim_mlp = 100 # this is the output dimension of mlp (a hyperparameter)

input_dim_graphsage = output_dim_mlp # input dimension of graphsage = output dimension of mlp
hidden_dim_graphsage = 75 #this is a hyperparameter
output_dim_graphsage = 50 #this is a hyperparameter

model_mlp = MLP(input_dim=in_dim_mlp, 
                 output_dim= output_dim_mlp)
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)



model_graphsage = GraphSAGE(in_dim=input_dim_graphsage, 
                 hidden_dim=hidden_dim_graphsage, 
                 out_dim= output_dim_graphsage)
optimizer_graphsage = torch.optim.Adam(model_graphsage.parameters(), lr=lr)

adj_2d = structure_adjacency_matrix()

for epoch in range(1, 1 + epochs):
    loss = train(model_mlp, optimizer_mlp, 
        model_graphsage, optimizer_graphsage,
        data_nodeLabels, adj_2d)
    print(loss)