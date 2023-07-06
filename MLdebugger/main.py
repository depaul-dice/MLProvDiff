import pickle, random, argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Preprocess import data
from CustomDataset import TraceDataset
from model import CombinedModel

def main(args):

    # get arguments
    file_name = args.file_name
    use_ratio = args.use_ratio
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    random.seed(args.random_seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    # load and process data
    num_features, feature_matrix, edge_list, traces_x, traces_y = data(file_name, use_ratio)

    # random split data into train and test
    random.shuffle(traces_x)
    random.shuffle(traces_y)
    num_train = int(len(traces_x)*0.2)
    train_x, train_y = traces_x[:num_train], traces_y[:num_train]
    test_x, test_y = traces_x[num_train:], traces_y[num_train:]

    train_loader = DataLoader(TraceDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(TraceDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_features, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # training and testing
    for epoch in range(num_epochs):
        # train
        for batch in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            trace_x, trace_y = batch
            trace_x, trace_y = trace_x.to(device), trace_y.to(device)
            combined = model(trace_x, feature_matrix, edge_list)
            combined = combined.view(-1, combined.size(-1))
            trace_y = trace_y.view(-1)
            loss = F.cross_entropy(combined, trace_y)
            loss.backward()
            optimizer.step()

        if 'loss' not in locals():
            raise Exception('No training data supplied. Check the amount!')

        # test
        model.eval()
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch in test_loader:
                trace_x, trace_y = batch
                combined = model(trace_x, feature_matrix, edge_list)
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
    args = parser.parse_args()

    main(args)
