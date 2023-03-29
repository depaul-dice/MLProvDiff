import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SGConv

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


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
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        #return torch.log_softmax(x, dim=-1)
        return x

class bilstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout= 0.2):
        """
        hidden_dim in bilstm should be output_dim in graphsage divided by 2
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, dropout = dropout)
        
    def forward(self, x):
        # x is a batch of input sequences
        # x has shape (batch_size, seq_len)
        
        # Embed the input sequence
        embedded_seq = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        
        # Pass the embedded sequence through the BiLSTM
        lstm_out, _ = self.bilstm(embedded_seq) # (batch_size, seq_len, 2 * hidden_dim)
        
        # Return the output of the BiLSTM as the embeddings
        return lstm_out

        
class MLP(torch.nn.Module):
    """
    input dim is vocab size
    output dim is same as input dim in graphsage (it is 200 right now)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 500)
        self.hidden_fc = nn.Linear(500,250)
        self.output_fc = nn.Linear(250, output_dim)

    def forward(self, x):
        # 100 * 24000
        # x = [num_nodes,vocab_size]
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [num_nodes, 500]

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [num_nodes, 250]

        output_mlp = self.output_fc(h_2)
        output_mlp = [num_nodes, output dim]

        # 24000 *200
        return output_mlp


def loss(out_graphsage, out_bisltm, path):
    #TODO: do dot product between out_graphsage and out_bilstm
    # and calculate the cross entropy loss between the above dot product 
    # and the path

def evaluator():
    print("Evaluating training and validation loss")


"""
BiLSTM trace will be trained in batches
but while each training, we need to train the entire graph and mlp at the with each batch
training
"""
def train(model_graphsage, data, train_idx, optimizer_graphsage, model_bilstm, optimizer_bilstm, data_traces):

    #for batch in mlp_dataloader
    model_mlp.train()
    optimizer_bilstm.zero_grad()
    out_mlp = model_mlp(data.x.feature)

    #200*150
    #TODO: set out_mlp as feature of graph data

    model_graphsage.train()
    optimizer_graphsage.zero_grad()
    out_graphsage = model_graphsage(data)[train_idx]
    #200*100

    # batch of traces 
    model_bilstm.train()
    optimizer_bilstm.zero_grad()
    out_bilstm = model_bilstm(data_traces.traces)
    
    #TODO: call loss function
    loss = 0
    return loss


data = "This is where graph data should be"
data_traces = "this should contain trace data and path data"


lr = 1e-4 
epochs = 50 
in_dim = 200 # this is the dimension of the feature vector for each node
hidden_dim = 75
embedding_dim = 500
output_dim = 100
#evaluator = Evaluator(name='ogbn-products')


vocab_size = 1000 #check later


model_mlp = mlp(in_dim=vocab_size, 
                 out_dim= in_dim)
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)



model_graphsage = GraphSAGE(in_dim=in_dim, 
                 hidden_dim=hidden_dim, 
                 out_dim= output_dim)
optimizer_graphsage = torch.optim.Adam(model_graphsage.parameters(), lr=lr)



model_bilstm = bilstm(vocab_size=vocab_size, 
                 embedding_dim = embedding_dim,
                 hidden_dim=output_dim/2)
optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=lr)



for epoch in range(1, 1 + epochs):
    loss = train(model, data, data_traces, train_idx, 
                model_mlp, optimizer_mlp,
                model_graphsage, optimizer_graphsage,
                model_bilstm, optimizer_bilstm)
    #result = test(model, data, split_idx, evaluator)
    print("epoch {} has loss={}".format(epoch,loss))