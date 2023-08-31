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

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_items_emb = self.item_embedding(pos_items.long())
        neg_items_emb = self.item_embedding(neg_items.long())
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=1)

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_items_emb.norm(2).pow(2) +
                          neg_items_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss


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

    def bpr_loss(self, users, pos_items, neg_items):
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


class UltraGCN(nn.Module):
    def __init__(self, args, dataset):
        super(UltraGCN, self).__init__()
        self.user_num = dataset.n_user
        self.item_num = dataset.m_item
        self.embedding_dim = args.dim
        self.w1 = 1e-7
        self.w2 = 1
        self.w3 = 1e-7
        self.w4 = 1

        self.negative_weight = 200
        self.gamma = 1e-4
        self.lambda_ = 1e-3

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = {'beta_uD': dataset.beta_uD, 'beta_iD': dataset.beta_iD}
        self.ii_constraint_mat = dataset.ii_constraint_mat
        self.ii_neighbor_mat = dataset.ii_neighbor_mat

        self.initial_weight = 1e-3
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(
                device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)

        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                      weight=omega_weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)],
                                                      reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(
            self.ii_neighbor_mat[pos_items].to(device))  # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def bpr_loss(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def forward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)

        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device
