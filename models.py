import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, subgraph_size):
        super(Model, self).__init__()
        self.W = nn.Linear(subgraph_size, subgra
        ph_size, bias=False)
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_in, n_h, activation)
        if readout == 'avg':
            self.read = AvgReadout()
        self.disc1 = Discriminator(n_h, negsamp_round)
        self.disc2 = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj1, adj2, simi_attr, sparse=False):
        h_1 = self.gcn1(seq1, adj1, sparse)
        h_2 = self.gcn2(seq1, adj2, sparse)
        # h_2 = self.gcn2(seq1, adj1, sparse)
        # 平均readout得到c
        c = self.read(h_1[:,:-1,:])
        # 将传入的simi_attr线性变化
        simi_attr = self.W(simi_attr)
        simi_attr = torch.sigmoid(simi_attr)
        # simi_attr = F.softmax(simi_attr, dim=2)
        c_ = torch.bmm(simi_attr, h_1[:, :-1, :]).squeeze()
        # 散点Embedding
        h_mv1 = h_1[:,-1,:]
        h_mv2 = h_2[:,-1,:]
        ret1 = self.disc1(c, h_mv1)
        ret2 = self.disc1(c_, h_mv2)
        return ret1, ret2