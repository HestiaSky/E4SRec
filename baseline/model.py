import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, args, dataset):
        super(BasicModel, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.n_user, self.m_item = dataset.n_user, dataset.m_item
        self.dim = args.dim
        self.user_embedding = nn.Embedding(self.n_user, self.dim)
        self.item_embedding = nn.Embedding(self.m_item, self.dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)


class VanillaMF(BasicModel):
    def __init__(self, args, dataset):
        super(VanillaMF, self).__init__(args, dataset)
        self.act = nn.Sigmoid()

    def forward(self, instances):
        users_emb = self.user_embedding(instances.long())
        items_emb = self.item_embedding.weight
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def loss_func(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_items_emb = self.item_embedding(pos_items.long())
        neg_items_emb = self.item_embedding(neg_items.long())
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=-1)
        users_emb = users_emb.unsqueeze(1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=-1)

        neg_labels = torch.zeros(neg_ratings.size()).to(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_ratings, neg_labels).mean(dim=-1)

        pos_labels = torch.ones(pos_ratings.size()).to(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_ratings, pos_labels)

        loss = pos_loss + neg_loss * neg_items.shape[1]

        # loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        # reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
        #                   pos_items_emb.norm(2).pow(2) +
        #                   neg_items_emb.norm(2).pow(2))/float(len(users))
        return loss


class LightGCN(BasicModel):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__(args, dataset)
        self.layers = 3
        self.dropout = args.dropout
        self.act = nn.Sigmoid()
        self.Graph = dataset.Graph
        self.mode = 'train'

    def __graph_dropout(self, x, dropout):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + (1 - dropout)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - dropout)
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __message_passing(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.mode == 'train':
            g = self.__graph_dropout(self.Graph, self.dropout)
        else:
            g = self.Graph
        for layer in range(self.layers):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items

    def loss_func(self, users, pos_items, neg_items):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        pos_items_emb = items_emb[pos_items.long()]
        neg_items_emb = items_emb[neg_items.long()]
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=1)

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(self.user_embedding(users.long()).norm(2).pow(2) +
                          self.item_embedding(pos_items.long()).norm(2).pow(2) +
                          self.item_embedding(neg_items.long()).norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def forward(self, instances):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[instances.long()]
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

