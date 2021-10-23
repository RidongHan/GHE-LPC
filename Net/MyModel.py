import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from .Embedding import My_Entity_Aware_Embedding
from .Encoder import MyPCNN_V
from .HierEncoder import HierEncoder

class MyHierRelLayer(nn.Module):
    def __init__(self, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False):
        super(MyHierRelLayer, self).__init__()
        self.hier_rel_gate = nn.Linear(linear_in, linear_out)
        self.hier_mlp_1 = nn.Linear(linear_out, mlp_hidden_size, bias=mlp_bias)
        self.hier_mlp_2 = nn.Linear(mlp_hidden_size, mlp_out, bias=mlp_bias)
        self.layer_norm = nn.LayerNorm(mlp_out)
        self.mlp_bias = mlp_bias
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.hier_rel_gate.weight) 
        nn.init.xavier_uniform_(self.hier_mlp_1.weight)
        nn.init.xavier_uniform_(self.hier_mlp_2.weight)
        nn.init.zeros_(self.hier_rel_gate.bias)
        if self.mlp_bias:
            nn.init.zeros_(self.hier_mlp_1.bias)
            nn.init.zeros_(self.hier_mlp_2.bias)

    def forward(self, S, hier_mat, hier_child_num=None, prev_layer_child_num=None):  
        # S [bs, h*3]
        "hier_rel"
        hier_logits = torch.matmul(S, hier_mat.t())
        hier_index = F.softmax(hier_logits, dim=-1)
        hier_relation = torch.matmul(hier_index, hier_mat)  # [bs, h*3],  relation-aware repre
        # hier_relation = torch.mean(hier_mat, dim=0).unsqueeze(0).repeat(S.shape[0], 1)
        "gate"
        concat_hier = torch.cat([S, hier_relation], dim=-1)
        alpha_hier = torch.sigmoid(self.hier_rel_gate(concat_hier))  # gate
        context_hier = alpha_hier * S + (1 - alpha_hier) * hier_relation  # relation-agument repre  , [bs, h*3]
        "MLP linear"
        middle_hier = F.relu(self.hier_mlp_1(context_hier))
        # middle_hier = F.relu(self.hier_mlp_1(concat_hier))
        output_hier = self.hier_mlp_2(middle_hier)  # [bs, h*3]
        "add&norm"
        output_hier += S  # [bs, h*3]
        output_hier = self.layer_norm(output_hier)
        # compute next_logits
        if hier_child_num is not None:
            tmp_list = []
            for i in range(len(hier_child_num)):
                tmp_list.append(hier_index[:, i:i+1].repeat_interleave(hier_child_num[i], dim=-1))
            next_logits = F.softmax(torch.cat(tmp_list, dim=-1), dim=-1)
        else:
            next_logits = None
        # compute prev_logits
        if prev_layer_child_num is not None:
            tmp_list = []
            i = 0
            j = 0
            while i < sum(prev_layer_child_num):
                tmp_list.append(torch.sum(hier_index[:, i:i+prev_layer_child_num[j]], dim=-1).unsqueeze(dim=1))
                i = i + prev_layer_child_num[j]
                j += 1
            prev_logits = torch.cat(tmp_list, dim=-1)
        else:
            prev_logits = None
                
        return hier_logits, output_hier, next_logits, prev_logits

class MyDenseNet(nn.Module):
    def __init__(self, linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False):
        super(MyDenseNet, self).__init__()
        self.hier1_rel_net = MyHierRelLayer(linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
        self.hier2_rel_net = MyHierRelLayer(linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)
        self.hier3_rel_net = MyHierRelLayer(linear_in, linear_out, mlp_hidden_size, mlp_out, mlp_bias=False)

    def forward(self, S, hier1_mat, hier2_mat, hier3_mat, hier1_child_num, hier2_child_num):
        hier1_logits, output_hier1, hier1_next_logits, hier1_prev_logits = self.hier1_rel_net(S, hier1_mat, hier1_child_num, None)
        hier2_logits, output_hier2, hier2_next_logits, hier2_prev_logits = self.hier2_rel_net(S, hier2_mat, hier2_child_num, hier1_child_num)
        hier3_logits, output_hier3, hier3_next_logits, hier3_prev_logits = self.hier3_rel_net(S, hier3_mat, None, hier2_child_num)
        return hier1_logits, output_hier1, hier1_next_logits, hier2_logits, output_hier2, hier2_next_logits, hier2_prev_logits, hier3_logits, output_hier3, hier3_prev_logits


class MyModel(nn.Module):
    def __init__(self, pre_word_vec, hier1_rel_num, hier2_rel_num, hier3_rel_num, lambda_embed=0.05, pos_dim=5, pos_len=100, hidden_size=230, dropout_rate=0.5, use_ghe=True, hier_encoder_heads=3):
        super(MyModel, self).__init__()
        word_embedding = torch.from_numpy(np.load(pre_word_vec))
        word_dim = word_embedding.shape[-1]
        self.hier1_rel_num = hier1_rel_num
        self.hier2_rel_num = hier2_rel_num
        self.hier3_rel_num = hier3_rel_num
        self.hidden_size = hidden_size
        # embedding
        self.embedding = My_Entity_Aware_Embedding(word_embedding, word_dim, lam=lambda_embed, pos_dim=pos_dim, pos_len=pos_len)
        # encoder
        input_embed_dim = 3 * word_dim
        self.PCNN = MyPCNN_V(input_embed_dim, hidden_size)
        # HierEncoder
        self.HierEncoder = HierEncoder(node_feat=3 * hidden_size, 
                                       node_hid=3 * hidden_size, 
                                       node_dropout=0.05, 
                                       alpha=0.2, 
                                       nheads=hier_encoder_heads,
                                       use_ghe=use_ghe)
        # hierarchical classifier chain
        self.dense_net = MyDenseNet(2 * 3 * hidden_size, 3 * hidden_size, 1024, 3 * hidden_size, mlp_bias=False)

        # selector
        combine_feature_dim = 3 * 3 * hidden_size
        self.bag_att_layer = nn.Linear(combine_feature_dim, 1, bias=False)

        # classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifer = nn.Linear(combine_feature_dim, hier3_rel_num)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifer.weight)
        nn.init.xavier_uniform_(self.bag_att_layer.weight)
        nn.init.zeros_(self.classifer.bias)

    def bag_att(self, output_hier, X_Scope):
        prob_bag = self.bag_att_layer(output_hier)
        last_dim = output_hier.shape[-1]
        tower_repre = []
        for s in X_Scope:
            prob = F.softmax(torch.reshape(prob_bag[s[0]:s[1]], shape=(1, -1)), dim=1)
            one_bag = torch.reshape(torch.matmul(prob, output_hier[s[0]:s[1]]), shape=(last_dim, ))
            tower_repre.append(one_bag)
        stack_repre = torch.stack(tower_repre, dim=0)
        return stack_repre

    def bag_mean(self, output_hier, X_Scope):
        tower_repre = []
        for s in X_Scope:
            tower_repre.append(torch.mean(output_hier[s[0]:s[1]], dim=0))
        stack_repre = torch.stack(tower_repre, dim=0)
        return stack_repre
  
    def forward(self, X, X_Pos1, X_Pos2, X_Order, Ent_Pos, X_Index1, X_Index2, X_Ent1, X_Ent2, X_Mask, X_Scope, X_length):
        # Embeding
        Xp, Xe, X = self.embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        # Encoder
        S = self.PCNN(X, X_Mask)  # [?, 690]

        # HierEncoder
        hier1_mat, hier2_mat, hier3_mat, hier1_child_num, hier2_child_num = self.HierEncoder(self.hier1_rel_num, self.hier2_rel_num, self.hier3_rel_num)

        # hierarchical classifier chain
        hier1_logits, output_hier1, hier1_next_logits, \
            hier2_logits, output_hier2, hier2_next_logits, hier2_prev_logits, \
                hier3_logits, output_hier3, hier3_prev_logits = \
                    self.dense_net(S, hier1_mat, hier2_mat, hier3_mat, hier1_child_num, hier2_child_num)

        # combine features
        output_hier = torch.cat([output_hier1, output_hier2, output_hier3], dim=1)

        X = self.bag_att(output_hier, X_Scope)
        # X = self.bag_mean(output_hier, X_Scope)
        # Classifier
        X = self.dropout(X)
        X = self.classifer(X)

        res = [X, hier1_logits, hier2_logits, hier3_logits, hier1_next_logits, hier2_next_logits, hier2_prev_logits, hier3_prev_logits]

        return res
 