import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def bipartite_graph_dataset(args):
    if args.dataset in ['ML1M', 'ML25M']:
        return MovieDataset(args)
    elif args.dataset in ['Gowalla', 'Yelp']:
        return POIDataset(args)


class BipartiteGraphDataset(Dataset):
    def __init__(self, args):
        super(BipartiteGraphDataset, self).__init__()
        self.device = args.device
        self.dataset = args.dataset
        self.neg_num = args.neg_num

        self.n_user = 0
        self.m_item = 0

        self.trainData, self.testData = None, None
        self.trainDataSize = 0

        self.UserItemNet = None
        self.allUPos, self.allIPos = None, None
        self.Graph = None
        self.u_sim_matrix, self.i_sim_matrix = None, None
        self.S, self.SS, self.uFeats, self.iFeats = None, None, None, None
        self.uLabel, self.iLabel = None, None
        self.U, self.I, self.tuLabel, self.tiLabel = None, None, None, None

    def uniform_sampling(self):
        users = np.random.randint(0, self.n_user, self.trainDataSize)
        allPos = self.allUPos
        S = []
        for i, user in enumerate(users):
            posForUser = allPos[user]
            while len(posForUser) == 0:
                user = np.random.randint(0, self.n_user)
                posForUser = allPos[user]
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        self.S = torch.LongTensor(S)

        items = np.random.randint(0, self.m_item, self.trainDataSize)
        allPos = self.allIPos
        SS = []
        for i, item in enumerate(items):
            posForItem = allPos[item]
            while len(posForItem) == 0:
                item = np.random.randint(0, self.m_item)
                posForItem = allPos[item]
            posindex = np.random.randint(0, len(posForItem))
            posuser = posForItem[posindex]
            while True:
                neguser = np.random.randint(0, self.n_user)
                if neguser in posForItem:
                    continue
                else:
                    break
            SS.append([item, posuser, neguser])
        self.SS = torch.LongTensor(SS)

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

    def get_sparse_graph(self, user_threshold, item_threshold):
        print("loading matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('datasets/' + self.dataset + '/s_pre_adj_mat.npz')
                u_sim_matrix = torch.load('datasets/' + self.dataset + '/user_sim_mat.pt')
                i_sim_matrix = torch.load('datasets/' + self.dataset + '/item_sim_mat.pt')
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
                sp.save_npz('datasets/' + self.dataset + '/s_pre_adj_mat.npz', norm_adj)

                print("generating similarity matrix")
                s = time.time()
                R = self.__convert_sp_mat_to_sp_tensor(self.UserItemNet).coalesce().to_dense().to(self.device)
                Rd = R.sum(dim=1).sqrt().unsqueeze(1)
                Rd[Rd == 0.] = 1.
                Ru = R / Rd
                RT = R.t()
                Rd = RT.sum(dim=1).sqrt().unsqueeze(1)
                Rd[Rd == 0.] = 1.
                Ri = RT / Rd
                u_sim_matrix = torch.mm(Ru, Ru.t())
                i_sim_matrix = torch.mm(Ri, Ri.t())
                torch.save(u_sim_matrix, 'datasets/' + self.dataset + '/u_sim_mat.pt')
                torch.save(i_sim_matrix, 'datasets/' + self.dataset + '/i_sim_mat.pt')
                print(f"costing {time.time()-s}s, saved sim_mat...")

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)
            u_sim_matrix = (u_sim_matrix > user_threshold).float() * u_sim_matrix
            i_sim_matrix = (i_sim_matrix > item_threshold).float() * i_sim_matrix
            Ud = u_sim_matrix.sum(dim=1, keepdim=True)
            Ud[Ud == 0.] = 1.
            Id = i_sim_matrix.sum(dim=1, keepdim=True)
            Id[Id == 0.] = 1.

            u_sim_matrix = u_sim_matrix / Ud
            i_sim_matrix = i_sim_matrix / Id
            self.u_sim_matrix, self.i_sim_matrix = u_sim_matrix, i_sim_matrix

        return self.Graph

    def __getitem__(self, idx):
        return self.S[idx]

    def __len__(self):
        return len(self.S)


class MovieDataset(BipartiteGraphDataset):
    def __init__(self, args):
        super(MovieDataset, self).__init__(args)

        self.trainData = pd.read_csv('../datasets/general/' + self.dataset + '/' + 'ntrain.txt', sep='\t',
                                     names=['UserID', 'MovieID'])
        self.testData = pd.read_csv('../datasets/general/' + self.dataset + '/' + 'ntest.txt', sep='\t',
                                    names=['UserID', 'MovieID'])
        self.n_user = max(self.trainData['UserID'].max(), self.testData['UserID'].max()) + 1
        self.m_item = max(self.trainData['MovieID'].max(), self.testData['MovieID'].max()) + 1
        self.trainDataSize = len(self.trainData)

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainData['UserID'], self.trainData['MovieID'])),
                                      shape=(self.n_user, self.m_item))
        self.allUPos = self.get_user_pos_items(list(range(self.n_user)))
        self.allIPos = self.get_item_pos_users(list(range(self.m_item)))
        self.testDict = self.__build_test()
        self.uLabel, self.iLabel = None, None
        self.get_labels()
        self.U, self.I, self.tuLabel, self.tiLabel = self.trainData['UserID'].copy().to_numpy(), \
                                                     self.trainData['MovieID'].copy().to_numpy(), None, None
        self.init_bathes()
        g = self.UserItemNet.tocoo()
        self.g = torch.sparse.FloatTensor(torch.from_numpy(np.vstack((g.row, g.col)).astype(np.int64)),
                                          torch.from_numpy(g.data),
                                          torch.Size(g.shape)).to_dense()
        self.items_D = self.g.sum(dim=0)
        self.users_D = self.g.sum(dim=1)
        self.beta_uD = (torch.sqrt(self.users_D + 1) / self.users_D).reshape(-1, 1)
        self.beta_iD = (1 / torch.sqrt(self.items_D + 1)).reshape(1, -1)
        self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat()

    def get_ii_constraint_mat(self):

        print('Computing \\Omega for the item-item graph... ')
        A = self.g.T.mm(self.g)
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, 10))
        res_sim_mat = torch.zeros((n_items, 10))
        items_D = A.sum(dim=0)
        users_D = A.sum(dim=1)

        beta_uD = (torch.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / torch.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = beta_uD.mm(beta_iD)
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * A[i]
            row_sims, row_idxs = torch.topk(row, 10)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()

    def __build_test(self):
        tdu, tdi = {}, {}
        for idx, row in self.testData.iterrows():
            user, item = row['UserID'], row['MovieID']
            tdu[user] = tdu.get(user, [])
            tdu[user].append(item)
            tdi[item] = tdi.get(item, [])
            tdi[item].append(user)
        return {'User': tdu, 'Item': tdi}


class POIDataset(BipartiteGraphDataset):
    def __init__(self, args):
        super(POIDataset, self).__init__(args)

        self.trainData = pd.read_csv('../datasets/general/' + self.dataset + '/' + self.dataset + '_train.txt', sep='\t',
                                     names=['UserID', 'ItemID', 'times'])
        self.tuneData = pd.read_csv('../datasets/general/' + self.dataset + '/' + self.dataset + '_tune.txt', sep='\t',
                                     names=['UserID', 'ItemID', 'times'])
        self.testData = pd.read_csv('../datasets/general/' + self.dataset + '/' + 'LT_test.txt', sep='\t',
                                     names=['UserID', 'ItemID', 'times'])

        self.n_user = max(self.trainData['UserID'].max(), self.tuneData['UserID'].max(), self.testData['UserID'].max()) + 1
        self.m_item = max(self.trainData['ItemID'].max(), self.tuneData['ItemID'].max(), self.testData['ItemID'].max()) + 1

        self.trainDataSize = len(self.trainData)

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainData['UserID'], self.trainData['ItemID'])),
                                      shape=(self.n_user, self.m_item))
        self.allUPos = self.get_user_pos_items(list(range(self.n_user)))
        self.allIPos = self.get_item_pos_users(list(range(self.m_item)))
        self.testDict = self.__build_test()
        self.uLabel, self.iLabel = None, None
        self.get_labels()
        self.U, self.I, self.tuLabel, self.tiLabel = self.trainData['UserID'].copy().to_numpy(), \
                                                     self.trainData['ItemID'].copy().to_numpy(), None, None
        self.init_bathes()

    def __build_test(self):
        tdu, tdi = {}, {}
        for idx, row in self.testData.iterrows():
            user, item = row['UserID'], row['ItemID']
            tdu[user] = tdu.get(user, [])
            tdu[user].append(item)
            tdi[item] = tdi.get(item, [])
            tdi[item].append(user)
        return {'User': tdu, 'Item': tdi}


