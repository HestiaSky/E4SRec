import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# ====================Metrics==============================
def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k+1)
    MRR = np.sum(pred / weight, axis=1) / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MRR = np.sum(MRR)
    return MRR


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k+1)
    AP = np.sum(pred * rank / weight, axis=1)
    AP = AP / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MAP = np.sum(AP)
    return MAP


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = pred * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg = np.sum(ndcg)
    return ndcg


def AUC(all_item_scores, dataset, test):
    r_all = np.zeros((dataset.m_item, ))
    r_all[test] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype('float')
# ====================end Metrics=============================



