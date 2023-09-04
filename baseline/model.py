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
        nn.init.normal_(self.user_embedding.weight, std=args.weight_decay)
        nn.init.normal_(self.item_embedding.weight, std=args.weight_decay)


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


class SASRec(nn.Module):
    def __init__(self, args, dataset):
        super(SASRec, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.m_item = dataset.m_item
        self.dim = args.dim
        self.dropout = args.dropout
        self.embedding = nn.Embedding(self.m_item, self.dim)
        self.pos_embedding = nn.Embedding(args.maxlen, self.dim)
        self.emb_dropout = nn.Dropout(p=self.dropout)

        self.attention_layernorms = nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.dim, eps=1e-8)

        for _ in range(args.layers):
            new_attn_layernorm = nn.LayerNorm(self.dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(self.dim, 1, self.dropout, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1),
                                          nn.Dropout(p=self.dropout),
                                          nn.ReLU(),
                                          nn.Conv1d(self.dim, self.dim, kernel_size=1),
                                          nn.Dropout(p=self.dropout))
            self.forward_layers.append(new_fwd_layer)

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.act = nn.Sigmoid()

    def log2feats(self, log_seq):
        seqs = self.embedding(log_seq)
        seqs *= self.dim ** 0.5
        positions = np.tile(np.array(range(log_seq.shape[1])), [log_seq.shape[0], 1])
        seqs += self.pos_embedding(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seq.cpu() == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            residuals = self.forward_layers[i](seqs.transpose(-1, -2)).transpose(-1, -2)
            seqs = seqs + residuals
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def loss_func(self, seq, pos, neg):
        log_feats = self.log2feats(seq)

        pos_embs = self.embedding(pos)
        neg_embs = self.embedding(neg)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def forward(self, seq):
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        items_emb = self.embedding.weight
        ratings = self.act(torch.matmul(final_feat, items_emb.t()))

        return ratings




