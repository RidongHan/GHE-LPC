import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle

#### GAT ####
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = Parameter(torch.empty(size=(in_features, out_features)))
        self.a = Parameter(torch.empty(size=(2*out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # LeakyRelu(x) = max(0, alpha * x)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.shape[0] # number of nodes
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # e1, ..., e1, e2, ..., e2, ... --> N times for each node
        Wh_repeated_alternating = Wh.repeat(N, 1)  # e1, e2, ..., en, e1, e2, ..., en, e1, ... --> N times for the squence
        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1, e1 || e2, ..., e1 || eN, ...
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  # [N * N, 2 * out_features]
        return all_combinations_matrix.view(N, N, 2 * self.out_features), Wh_repeated_in_chunks, Wh_repeated_alternating

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input, Wh_repeated_in_chunks, Wh_repeated_alternating = self._prepare_attentional_mechanism_input(Wh)
        # Linear Layer
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]

        return h_prime


class GAT(nn.Module):
    def __init__(self, node_feat, node_hid, dropout, alpha, nheads, concat=False):
        """Dense/multi-head version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.concat = concat

        self.attentions = [GraphAttentionLayer(node_feat, node_hid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:  # multi-heads 
            y = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            y = torch.mean(torch.stack([att(x, adj) for att in self.attentions]), dim=0)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.elu(y)  # elu(x) = max(0, x) + min(0, alpha * (exp(x)-1))
        return y

#### HierEncoder ####
class HierEncoder(nn.Module):
    def __init__(self, node_feat=690, node_hid=690, node_dropout=0.05, alpha=0.2, nheads=3, use_ghe=True):
        """
        Hierarchies Encoder module
        :param node_feat:
        :param node_hid:
        :param node_dropout:
        """
        super(HierEncoder, self).__init__()
        self.use_ghe = use_ghe

        self.label_map = self.get_label_map("./data/tree/rel2id_tree.txt")
        self.label_embedding = nn.Embedding(len(self.label_map), node_feat)

        self.adj_matrix, self.node_child_num = self.get_adj_matrix("./data/tree/nyt_tree.taxonomy", self.label_map)

        if self.use_ghe:
            self.model = GAT(node_feat=node_feat, node_hid=node_hid, dropout=node_dropout, alpha=alpha, nheads=nheads)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.label_embedding.weight)

    def get_label_map(self, label2id_file):
        label_map = {}
        f = open(label2id_file)
        num_rel = int(f.readline().strip())
        for line in f.readlines():
            line = line.strip().split()
            label_map[line[0]] = int(line[1])
        assert(num_rel == len(label_map))
        return label_map

    def get_adj_matrix(self, hierar_taxonomy, label_map):
        """
        get adj_matrix from given hierar_taxonomy
        :param hierar_taxonomy: Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        """
        node_child_num = {}
        adj_matrix = torch.eye(len(label_map), dtype=torch.long).cuda()
        with open(hierar_taxonomy) as f:
            for line in f:
                line_split = line.rstrip().split()
                parent_label, children_label = line_split[0], line_split[1:]
                parent_label_id = label_map[parent_label]
                children_label_ids = [label_map[c] for c in children_label]
                node_child_num[parent_label_id] = len(children_label_ids)
                for c in children_label_ids:
                    adj_matrix[parent_label_id][c] = 1
                    adj_matrix[c][parent_label_id] = 1

        return adj_matrix, node_child_num

    def get_label_representation(self):
        """
        get output of each node as the structure-aware label representation
        """
        if self.use_ghe:
            label_embed = self.label_embedding(torch.LongTensor(torch.arange(0, len(self.label_map))).cuda())
            tree_label_feature = self.model(label_embed, self.adj_matrix)
            return tree_label_feature
        else:
            label_embed = self.label_embedding(torch.LongTensor(torch.arange(0, len(self.label_map))).cuda())
            return label_embed

    def forward(self, hier1_num, hier2_num, hier3_num):
        label_feature = self.get_label_representation()
        na = label_feature[0:1]
        hier1 = label_feature[0: hier1_num] 
        hier2 = torch.cat([na, label_feature[hier1_num: hier1_num+hier2_num-1]], dim=0)
        hier3 = torch.cat([na, label_feature[hier1_num+hier2_num-1: -1]], dim=0)

        child_num_hier1 = [1]
        child_num_hier2 = [1]
        for i in range(1, hier1_num):
            child_num_hier1.append(self.node_child_num[i])
        for i in range(hier1_num, hier1_num + hier2_num-1):
            child_num_hier2.append(self.node_child_num[i])

        return hier1, hier2, hier3, child_num_hier1, child_num_hier2
