import argparse
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from eval_utils import *
from data_utils import bipartite_graph_dataset
from model import *


MODEL = {'VanillaMF': VanillaMF, 'LightGCN': LightGCN, 'UltraGCN': UltraGCN}


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': 0,
        'epochs': 100,
        'weight_decay': 0,
        'seed': 42,
        'model': 'UltraGCN',
        'dim': 128,
        'layers': 2,
        'dataset': 'ML1M',
        'topk': [5, 10, 20],
        'patience': 5,
        'eval_freq': 5,
        'lr_reduce_freq': 100,
        'save_freq': 1,
        'neg_num': -1,
        'batch_size': 1024,
        'gamma': 0.5,
        'save': 0,
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
torch.autograd.set_detect_anomaly(True)

dataset = bipartite_graph_dataset(args)
model = MODEL[args.model](args, dataset)
print(str(model))

optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_reduce_freq, gamma=float(args.gamma))
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)


def train():
    model.train()
    model.mode = 'train'
    t = time.time()
    dataset.init_bathes()
    avg_loss = 0.
    batch_num = dataset.trainDataSize // args.batch_size + 1
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.trainDataSize \
            else torch.arange(i * args.batch_size, dataset.trainDataSize)
        users, items, user_labels, item_labels = torch.LongTensor(dataset.U[indices]).to(args.device), \
                                                 torch.LongTensor(dataset.I[indices]).to(args.device),\
                                                 torch.FloatTensor(dataset.tuLabel[indices]).to(args.device),\
                                                 torch.FloatTensor(dataset.tiLabel[indices]).to(args.device)

        ratings = model(users)
        optimizer.zero_grad()
        loss = model.bpr_loss(ratings, user_labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().item()

    lr_scheduler.step()
    avg_loss /= dataset.trainDataSize // args.batch_size + 1
    print(f'Average loss:{avg_loss} \n Epoch time: {time.time()-t} \n')


def test():
    model.eval()
    model.mode = 'test'
    testDict = dataset.testDict
    with torch.no_grad():
        model.build_graph()
        users, items = list(testDict['User'].keys()), list(testDict['Item'].keys())

        results = {'Precision': np.zeros(len(args.topk)),
                   'Recall': np.zeros(len(args.topk)),
                   'MRR': np.zeros(len(args.topk)),
                   'MAP': np.zeros(len(args.topk)),
                   'NDCG': np.zeros(len(args.topk))}
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i*args.batch_size: (i+1)*args.batch_size] \
                if (i+1)*args.batch_size <= len(users) else users[i*args.batch_size:]
            all_pos = dataset.get_user_pos_items(batch_users)
            groundTruth = [testDict['User'][u] for u in batch_users]
            batch_users = torch.LongTensor(batch_users).to(args.device)

            ratings = model(batch_users)
            exclude_index = []
            exclude_items = []
            for range_i, its in enumerate(all_pos):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)
            ratings[exclude_index, exclude_items] = -(1 << 10)
            _, ratings_K = torch.topk(ratings, k=args.topk[-1])
            ratings_K = ratings_K.cpu().numpy()

            r = getLabel(groundTruth, ratings_K)
            for j, k in enumerate(args.topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                mrr = MRR_atK(groundTruth, r, k)
                map = MAP_atK(groundTruth, r, k)
                ndcg = NDCG_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += mrr
                results['MAP'][j] += map
                results['NDCG'][j] += ndcg

        for key in results.keys():
            results[key] /= float(len(users))
        print(f'Evaluation for User: \n')
        for j, k in enumerate(args.topk):
            print(f'Precision@{k}: {results["Precision"][j]} \n '
                  f'Recall@{k}: {results["Recall"][j]} \n '
                  f'MRR@{k}: {results["MRR"][j]} \n '
                  f'MAP@{k}: {results["MAP"][j]} \n '
                  f'NDCG@{k}: {results["NDCG"][j]} \n')


# Train Model
t_total = time.time()
for epoch in range(args.epochs):
    print(f'Epoch {epoch}')
    train()
    torch.cuda.empty_cache()
    if (epoch + 1) % args.eval_freq == 0:
        test()
        torch.cuda.empty_cache()
if args.save == 1:
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, 'datasets/' + args.dataset + '/' + args.model + '.pth')
    torch.cuda.empty_cache()

print(f'Model training finished! Total time is {time.time()-t_total}')





