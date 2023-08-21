import pickle, random, argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from preprocess import data
from CustomDataset import TraceDataset
from model import CombinedModel
from transformerScheduler import TransformerScheduler

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

    train_loader = DataLoader(TraceDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    test_loader = DataLoader(TraceDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_features, num_features, num_layers=num_layers, encoder=encoder, num_heads=num_heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, warmup_steps=4000, d_model=num_features)

    # training and testing
    for epoch in range(num_epochs):
        # train
        for batch in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            trace_x, trace_y = batch
            trace_x, trace_y, feature_matrix, edge_list = trace_x.to(device), trace_y.to(device), feature_matrix.to(device), edge_list.to(device)
            combined = model(trace_x, feature_matrix, edge_list)
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
        model.eval()
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch in test_loader:
                trace_x, trace_y = batch
                trace_x, trace_y, feature_matrix, edge_list = trace_x.to(device), trace_y.to(device), feature_matrix.to(device), edge_list.to(device)
                combined = model(trace_x, feature_matrix, edge_list) # B * T * N

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
