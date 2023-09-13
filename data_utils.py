import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollator
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class BipartiteGraphDataset(Dataset):
    def __init__(self, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.dataset = dataset

        self.trainData, self.allPos, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0
        with open('datasets/general/' + self.dataset + '/train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.allPos[user] = items
                for item in items:
                    self.trainData.append([user, item])
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        with open('datasets/general/' + self.dataset + '/test.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.testData[user] = items
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1
        # self.uLabel = None
        # self.get_labels()
        # self.U, self.tuLabel = list(self.allPos.keys()), None

    # def get_labels(self):
    #     u_label = []
    #     allUPos = self.allPos
    #     for user in range(self.n_user):
    #         user_pos = allUPos[user]
    #         user_label = torch.zeros(self.m_item, dtype=torch.float)
    #         user_label[user_pos] = 1.
    #         # user_label = 0.9 * user_label + (1.0 / self.m_item)
    #         u_label.append(user_label.tolist())
    #     self.uLabel = torch.FloatTensor(u_label)
    #
    # def init_bathes(self):
    #     np.random.shuffle(self.U)
    #     self.tuLabel = self.uLabel[self.U]

    def __getitem__(self, idx):
        user, item = self.trainData[idx]
        return user, self.allPos[user], item

    def __len__(self):
        return len(self.trainData)


class BipartiteGraphCollator(DataCollator):
    def __call__(self, batch) -> dict:
        user, items, labels = zip(*batch)
        max_len = max([len(item) for item in items])
        inputs = [[user] + [0] * (max_len - len(item)) + item for item in items]
        inputs_mask = [[1] + [0] * (max_len - len(item)) + [1] * len(item) for item in items]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }


class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen):
        super(SequentialDataset, self).__init__()
        self.dataset = dataset
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], [], []
        self.n_user, self.m_item = 0, 0

        with open('datasets/sequential/' + self.dataset + '/' + self.dataset + '.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))
                if len(items) >= 3:
                    train_items = items[:-2]
                    length = min(len(train_items), self.maxlen)
                    for t in range(length):
                        self.trainData.append([train_items[:-length + t], train_items[-length + t]])
                    self.valData.append([items[:-2], items[-2]])
                    self.testData.append([items[:-1], items[-1]])
                else:
                    for t in range(len(items)):
                        self.trainData.append([items[:-len(items) + t], items[-len(items) + t]])
                    self.valData.append([])
                    self.testData.append([])

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

        self.allPos = {}
        with open('datasets/sequential/' + self.dataset + '/' + self.dataset + '_sample.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.allPos[user] = items

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user] + self.testData[user])
        return posItems

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)


class SequentialCollator(DataCollator):
    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }

