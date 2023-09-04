import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class BipartiteGraphDataset(Dataset):
    def __init__(self, args):
        super(BipartiteGraphDataset, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.neg_num = args.neg_num

        self.trainData, self.testData = [], []
        self.allUPos = []
        self.users, self.pos_items, self.neg_items = None, None, None

        with open('../datasets/general/' + self.dataset + '/train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) for item in line[1:]]
                self.allUPos.append(items)
                for item in items:
                    self.trainData.append([user, item])

        with open('../datasets/general/' + self.dataset + '/test.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) for item in line[1:]]
                for item in items:
                    self.testData.append([user, item])

        self.trainData, self.testData = np.array(self.trainData), np.array(self.testData)
        self.trainDataSize = len(self.trainData)
        self.n_user = max(self.trainData[:, 0].max(), self.testData[:, 0].max()) + 1
        self.m_item = max(self.trainData[:, 1].max(), self.testData[:, 1].max()) + 1

        self.testDict = self.__build_test()
        self.uLabel = None
        self.get_labels()
        self.U, self.I, self.tuLabel = self.trainData[:, 0], self.trainData[:, 1], None

    def negative_sampling(self, batch_size, neg_num):
        neg_items = np.random.randint(0, self.m_item, (batch_size, neg_num))
        neg_items = torch.LongTensor(neg_items)
        return neg_items

    def get_labels(self):
        u_label = []
        allUPos = self.allUPos
        for user in range(self.n_user):
            user_pos = allUPos[user]
            user_label = torch.zeros(self.m_item, dtype=torch.float)
            user_label[user_pos] = 1.
            user_label = 0.9 * user_label + (1.0 / self.m_item)
            u_label.append(user_label.tolist())
        self.uLabel = torch.FloatTensor(u_label)

    def init_bathes(self):
        np.random.shuffle(self.U)
        self.tuLabel = self.uLabel[self.U]

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allUPos[user])
        return posItems

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float64)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('datasets/general/' + self.dataset + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float64)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz('datasets/general/' + self.dataset + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

        return self.Graph

    def __build_test(self):
        tdu = {}
        for user, item in self.testData:
            tdu[user] = tdu.get(user, [])
            tdu[user].append(item)
        return tdu

    def __getitem__(self, idx):
        return self.trainData[idx]

    def __len__(self):
        return self.trainDataSize


class SequentialDataset(Dataset):
    def __init__(self, args):
        super(SequentialDataset, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.maxlen = args.maxlen

        self.trainData, self.valData, self.testData = [], [], []
        self.n_user, self.m_item = 0, 0

        with open('../datasets/sequential/' + self.dataset + '/' + self.dataset + '.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))
                if len(items) >= 3:
                    self.trainData.append(items[:-2])
                    self.valData.append([items[-2]])
                    self.testData.append([items[-1]])
                else:
                    self.trainData.append(items)
                    self.valData.append([])
                    self.testData.append([])

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

        self.allPos = {}
        with open('../datasets/sequential/' + self.dataset + '/' + self.dataset + '_sample.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.allPos[user] = items

    def negative_sampling(self, left, right, ts):
        t = np.random.randint(left, right)
        while t in ts:
            t = np.random.randint(left, right)
        return t

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user] + self.testData[user])
        return posItems

    def test_seq_generate(self, idx, subset='test'):
        seqs = []
        for i in idx:
            if len(self.valData[i]) < 1 or len(self.testData[i]) < 1:
                continue
            seq = np.zeros([self.maxlen], dtype=np.int32)
            tmp = self.maxlen - 1
            if subset == 'test':
                seq[tmp] = self.valData[i][0]
                tmp -= 1
            for t in reversed(self.trainData[i][:-1]):
                seq[tmp] = t
                tmp -= 1
                if tmp == -1:
                    break
            seqs.append(seq)
        seqs = np.array(seqs)
        return seqs

    def __getitem__(self, idx):
        seq, pos, neg = (np.zeros([self.maxlen], dtype=np.int32),
                         np.zeros([self.maxlen], dtype=np.int32),
                         np.zeros([self.maxlen], dtype=np.int32))
        nxt = self.trainData[idx][-1]
        tmp = self.maxlen - 1
        ts = set(self.trainData[idx])
        for t in reversed(self.trainData[idx][:-1]):
            seq[tmp] = t
            pos[tmp] = nxt
            if nxt != 0:
                neg[tmp] = self.negative_sampling(1, self.m_item, ts)
            nxt = t
            tmp -= 1
            if tmp == -1:
                break
        return seq, pos, neg

    def __len__(self):
        return len(self.trainData)



