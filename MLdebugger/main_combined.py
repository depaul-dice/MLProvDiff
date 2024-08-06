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
    train_file_names = args.train_file_names
    test_file_names = args.test_file_names
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

    train_file_names = train_file_names.split(",")
    test_file_names = test_file_names.split(",")

    # print arguments
    print('train sets: ', train_file_names)
    print('test sets: ', test_file_names)
    print('use trained data for test: ', use_train)
    print('use mask: ', use_mask)
    print('gpu: ', torch.cuda.is_available())
    print('devices: ', torch.cuda.device_count())
    # empty cuda cache
    with torch.no_grad():
      torch.cuda.empty_cache()
    gc.collect()

    # load and process train data
    num_features, feature_matrix_train, edge_list_train, train_x, train_y = data(train_file_names[0], use_ratio)
    for train_file_name in train_file_names[1:]:
      num_features, feature_matrix_i, edge_list_i, train_x_i, train_y_i = data(train_file_name, use_ratio)
      feature_matrix_train = torch.cat((feature_matrix_train, feature_matrix_i))
      edge_list_train = torch.cat((edge_list_train, edge_list_i), dim=1)
      train_x = torch.cat((train_x, train_x_i), dim=1)
      train_y = torch.cat((train_y, train_y_i), dim=1)

    # load and process test data
    num_features, feature_matrix_test, edge_list_test, test_x, test_y = data(test_file_names[0], use_ratio)
    for test_file_name in test_file_names[1:]:
      num_features, feature_matrix_i, edge_list_i, test_x_i, test_y_i = data(test_file_name, use_ratio)
      print(feature_matrix_test.size())
      feature_matrix_test = torch.cat((feature_matrix_test, feature_matrix_i))
      edge_list_test = torch.cat((edge_list_test, edge_list_i), dim=1)
      test_x = torch.cat((test_x, test_x_i), dim=1)
      test_y = torch.cat((test_y, test_y_i), dim=1)
    # random split data into train and test
    # temp = list(zip(traces_x, traces_y))
    # random.shuffle(temp)
    # traces_x, traces_y = zip(*temp)

    # num_train = int(len(traces_x)*0.8)
    # if use_train: # use all data for training
    #     train_x, train_y = traces_x, traces_y
    # else:
    #     train_x, train_y = traces_x[:num_train], traces_y[:num_train]
    # test_x, test_y = traces_x[num_train:], traces_y[num_train:]
    
    train_loader = DataLoader(TraceDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(TraceDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
#    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
#    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    device = 'cpu'
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            trace_x, trace_y, feature_matrix_train, edge_list_train = trace_x.to(device2), trace_y.to(device3), feature_matrix_train.to(device1), edge_list_train.to(device1)
            embeddings = model1(feature_matrix_train, edge_list_train)
            embeddings = embeddings.to(device2)
            out_encoder = model2(trace_x)
            out_encoder = out_encoder.to(device3)
            embeddings = embeddings.to(device3)
            combined = model3(out_encoder, embeddings)
            combined = combined.view(-1, combined.size(-1))
            trace_y = trace_y.view(-1)
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
                trace_x, trace_y, feature_matrix_test, edge_list_test = trace_x.to(device2), trace_y.to(device3), feature_matrix_test.to(device1), edge_list_test.to(device1)
                embeddings = model1(feature_matrix_test, edge_list_test)
                embeddings = embeddings.to(device2)
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
                num_correct += pred.eq(trace_y).sum()
                num_total += len(trace_y.view(-1))

        print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}'.format(epoch, loss, float(num_correct) / num_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_names', type=str)
    parser.add_argument('--test_file_names', type=str)
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