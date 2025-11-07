import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl

import argparse
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='ANIMA: Addressing the Distortion of Community Representations in Anomaly Detection on Attributed Network')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--eps', type=float)


args = parser.parse_args()
if args.lr is None:
    if args.dataset in ['cora']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['cora']:
        args.num_epoch = 100

if args.eps is None:
    if args.dataset in ['cora']:
        args.eps = 0.1

if args.dataset == 'cora':
    lam = 0.2
    gamma = 0.9

batch_size = args.batch_size
subgraph_size = args.subgraph_size
print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, idx_train, idx_val,\
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

features, _ = preprocess_features(features)
dgl_graph = adj_to_dgl_graph(adj)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj_s = adj.todense()
adj_s = torch.FloatTensor(adj_s[np.newaxis])
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
aff_matri = s_dot(features[0])
aff_matri = aff_matri.unsqueeze(dim=0)
adj_hat = get_a_hat(adj_s[0].int(), aff_matri, args.eps).cpu()
adj_hat = normalize_adj(adj_hat[0])
adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
adj = torch.FloatTensor(adj[np.newaxis])
adj_hat = torch.FloatTensor(adj_hat[np.newaxis])
aff_matri = torch.FloatTensor(aff_matri)


# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.subgraph_size)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_hat = adj_hat.cuda()
    adj_s = adj_s.cuda()
    aff_matri = aff_matri.cuda()

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:, -1, :] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))

if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.cuda()
    added_adj_zero_col = added_adj_zero_col.cuda()
    added_feat_zero_row = added_feat_zero_row.cuda()

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')

    for epoch in range(args.num_epoch):
        loss_full_batch = torch.zeros((nb_nodes,1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.cuda()

        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.
        similar_degree = []
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            ba_hat = []
            bf = []
            bs = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
            if torch.cuda.is_available():
                lbl = lbl.cuda()
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]

                cur_feat = features[:, subgraphs[i], :]
                cur_simi = aff_matri[:, i, subgraphs[i]]
                ba.append(cur_adj)
                ba_hat.append(cur_adj_hat)
                bf.append(cur_feat)
                bs.append(cur_simi)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            ba_hat = torch.cat(ba_hat)
            ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
            ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)
            bs = torch.cat(bs)
            bs = bs.unsqueeze(dim=1)

            logits1, logits2 = model(bf, ba, ba_hat, bs)
            loss_all = (1-gamma) * b_xent(logits1, lbl) + gamma * b_xent(logits2, lbl)
            loss = torch.mean(loss_all)
            # print(loss)
            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model.pkl')
        else:
            cnt_wait += 1

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)


# Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_model.pkl'))

multi_round_ano_score1 = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score2 = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            ba_hat = []
            bf = []
            bs = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.cuda()
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                cur_simi = aff_matri[:, i, subgraphs[i]]
                ba.append(cur_adj)
                ba_hat.append(cur_adj_hat)
                bf.append(cur_feat)
                bs.append(cur_simi)
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            ba_hat = torch.cat(ba_hat)
            ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
            ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bs = torch.cat(bs)
            bs = bs.unsqueeze(dim=1)

            with torch.no_grad():
                logits1, logits2 = model(bf, ba, ba_hat, bs)
                logits1 = torch.squeeze(logits1)
                logits1 = torch.sigmoid(logits1)
                logits2 = torch.squeeze(logits2)
                logits2 = torch.sigmoid(logits2)

            ano_score1 = - (logits1[:cur_batch_size] - logits1[cur_batch_size:]).cpu().numpy()
            ano_score2 = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()

            multi_round_ano_score1[round, idx] = ano_score1
            multi_round_ano_score2[round, idx] = ano_score2

        pbar_test.update(1)
ano_score_final1 = np.mean(multi_round_ano_score1, axis=0)
ano_score_final2 = np.mean(multi_round_ano_score2, axis=0)
s = s_dot(features[0])
a_hat = get_a_hat(adj_s[0].int(), s, args.eps)
aff_score = get_structual_score(adj_s[0], a_hat)
aff_score = aff_score.cpu().numpy()
ano_score_final = (1 - lam) * ((1 - gamma) * ano_score_final1 + gamma * ano_score_final2) + lam * aff_score
auc = roc_auc_score(ano_label, ano_score_final)
print('AUC:{:.4f}'.format(auc))




