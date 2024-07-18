
import pickle, random, argparse
from tqdm import tqdm
import gc

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import DataParallel

from Preprocess import data
from CustomDataset import TraceDataset
from model import GraphSAGE, Model2, CombinedModel
from transformerScheduler import TransformerScheduler

#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def main(args):

    # get arguments
    file_name = args.file_name
    use_ratio = args.use_ratio
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    use_train = args.use_train
    use_mask = args.use_mask
    num_layers = args.num_layers
    encoder = args.encoder
    num_heads = args.num_heads
    dropout = args.dropout
    random.seed(args.random_seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    # print arguments
    print('use trained data for test: ', use_train)
    print('use mask: ', use_mask)
    print('gpu: ', torch.cuda.is_available())
    print('devices: ', torch.cuda.device_count())
    # empty cuda cache
    with torch.no_grad():
      torch.cuda.empty_cache()
    gc.collect()

    # load and process data
    num_features, feature_matrix, edge_list, traces_x, traces_y = data(file_name, use_ratio)

    # random split data into train and test
    temp = list(zip(traces_x, traces_y))
    random.shuffle(temp)
    traces_x, traces_y = zip(*temp)

    num_train = int(len(traces_x)*0.8)
    if use_train: # use all data for training
        train_x, train_y = traces_x, traces_y
    else:
        train_x, train_y = traces_x[:num_train], traces_y[:num_train]
    test_x, test_y = traces_x[num_train:], traces_y[num_train:]
    
    train_loader = DataLoader(TraceDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(TraceDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
#    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
#    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    device = 'cpu'
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("device1: ", device1)
    print("device2: ", device2)
    print("device3: ", device3)

    hidden_dim = num_features if encoder == 'transformer' else hidden_dim
    model1 = GraphSAGE(num_features, hidden_dim)
    model1.to(device1)
    model2 = Model2(num_features, hidden_dim, num_layers=num_layers, encoder=encoder, num_heads=num_heads, dropout=dropout)
    model2.to(device2)
    model3 = CombinedModel()
    model3.to(device3)
#    model = CombinedModel(num_features, hidden_dim, num_layers=num_layers, encoder=encoder, num_heads=num_heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(list(model1.parameters())+list(model2.parameters())+list(model3.parameters()), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, warmup_steps=4000, d_model=num_features)

    # training and testing
    for epoch in range(num_epochs):
#        with torch.no_grad():
#            torch.cuda.empty_cache()
#        gc.collect()
        # train
        for batch in tqdm(train_loader):
            model1.train()
            model2.train()
            model3.train()
            optimizer.zero_grad()
            trace_x, trace_y = batch            
            trace_x_0, trace_x_1 = trace_x
            trace_y_0, trace_y_1 = trace_y
            trace_x_0, trace_x_1, trace_y_0, trace_y_1, feature_matrix, edge_list = trace_x_0.to(device2), trace_x_1.to(device2), \
            trace_y_0.to(device3), trace_y_1.to(device3), feature_matrix.to(device1), edge_list.to(device1)

            embeddings = model1(feature_matrix, edge_list)
            embeddings = embeddings.to(device2)
            
            trace_x = torch.cat((trace_x_0, trace_x_1))
            out_encoder = model2(trace_x)
            out_encoder = out_encoder.to(device3)
            embeddings = embeddings.to(device3)
            combined = model3(out_encoder, embeddings)
            pred = combined.argmax(dim=2)
            combined = combined.view(-1, combined.size(-1))
            trace_y = torch.cat((trace_y_0, trace_y_1))
            trace_y = trace_y.view(-1)
            import torchmetrics
            from torchmetrics.text import EditDistance
            metric = EditDistance()
            print(metric([pred], [trace_y]))
            exit()
            loss = F.cross_entropy(combined, trace_y)
            loss.backward()
            optimizer.step()
            if encoder == 'transformer':
                scheduler.step()
                
        if 'loss' not in locals():
            raise Exception('No training data supplied. Check the amount!')

        # test
        model1.eval()
        model2.eval()
        model3.eval()
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch in test_loader:
                trace_x, trace_y = batch
                trace_x_0, trace_x_1 = trace_x
                trace_y_0, trace_y_1 = trace_y
                trace_x_0, trace_x_1, trace_y_0, trace_y_1, feature_matrix, edge_list = trace_x_0.to(device2), trace_x_1.to(device2), \
                trace_y_0.to(device3), trace_y_1.to(device3), feature_matrix.to(device1), edge_list.to(device1)                
                embeddings = model1(feature_matrix, edge_list)
                embeddings = embeddings.to(device2)
                trace_x = torch.cat((trace_x_0, trace_x_1))
                out_encoder = model2(trace_x)
                out_encoder = out_encoder.to(device3)
                embeddings = embeddings.to(device3)
                combined = model3(out_encoder, embeddings) # B * T * N

                # masking
                if use_mask:
                    tx = trace_x.unsqueeze(2).expand(-1, -1, feature_matrix.size(0), -1) # B * T * F ->  B * T * N * F
                    fm = feature_matrix.unsqueeze(0).unsqueeze(0).expand(trace_x.size(0), trace_x.size(1), -1, -1) # N * F -> B * T * N * F
                    mask = (tx == fm).all(dim=-1) # B * T * N
                    combined = combined.masked_fill(~mask, float('-inf'))

                 # inference
                pred = combined.argmax(dim=2)
                trace_y = torch.cat((trace_y_0, trace_y_1))
                num_correct += pred.eq(trace_y).sum()
                num_total += len(trace_y.view(-1))

        print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, float(num_correct) / num_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='cat')
    parser.add_argument('--use_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--use_train', type=bool, default=False, help='use all data for training')
    parser.add_argument('--use_mask', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder', type=str, default='transformer')
    args = parser.parse_args()

    main(args)
