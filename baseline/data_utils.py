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
        self.UserItemNet = None
        self.allUPos, self.allIPos = None, None
        self.users, self.pos_items, self.neg_items = None, None, None
        self.uLabel, self.iLabel = None, None
        self.U, self.I, self.tuLabel, self.tiLabel = None, None, None, None

        with open('../datasets/general/' + self.dataset + '/train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) for item in line[1:]]
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

        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainData[:, 0], self.trainData[:, 1])),
            shape=(self.n_user, self.m_item))
        self.allUPos = self.get_user_pos_items(list(range(self.n_user)))
        self.allIPos = self.get_item_pos_users(list(range(self.m_item)))
        self.testDict = self.__build_test()
        self.uLabel, self.iLabel = None, None
        self.get_labels()
        self.U, self.I, self.tuLabel, self.tiLabel = self.trainData[:, 0], self.trainData[:, 1], None, None

    def negative_sampling(self, batch_size, neg_num):
        neg_items = np.random.randint(0, self.m_item, (batch_size, neg_num))
        neg_items = torch.LongTensor(neg_items)
        return neg_items

    def get_labels(self):
        u_label, i_label = [], []
        allUPos, allIPos = self.allUPos, self.allIPos
        for user in range(self.n_user):
            user_pos = allUPos[user]
            user_label = torch.zeros(self.m_item, dtype=torch.float)
            user_label[user_pos] = 1.
            user_label = 0.9 * user_label + (1.0 / self.m_item)
            u_label.append(user_label.tolist())
        for item in range(self.m_item):
            item_pos = allIPos[item]
            item_label = torch.zeros(self.n_user, dtype=torch.float)
            item_label[item_pos] = 1.
            item_label = 0.9 * item_label + (1.0 / self.n_user)
            i_label.append(item_label.tolist())
        self.uLabel, self.iLabel = torch.FloatTensor(u_label), torch.FloatTensor(i_label)

    def init_bathes(self):
        np.random.shuffle(self.U)
        np.random.shuffle(self.I)
        self.tuLabel = self.uLabel[self.U]
        self.tiLabel = self.iLabel[self.I]

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def get_item_pos_users(self, items):
        posUsers = []
        for item in items:
            posUsers.append(self.UserItemNet.T[item].nonzero()[1])
        return posUsers

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


