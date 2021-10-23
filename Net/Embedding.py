import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

class My_Entity_Aware_Embedding(nn.Module):
    def __init__(self, word_embedding, word_dim, lam=0.05, pos_dim=5, pos_len=100):
        super(My_Entity_Aware_Embedding, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
        self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.lam = lam
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2):
        X = self.word_embedding(X)
        Xp = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        Xe = self.word_ent_embedding(X, X_Ent1, X_Ent2)
        # gate
        A = torch.sigmoid(self.fc1(Xe / self.lam))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        return Xp, Xe, X

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        return torch.cat([X, X_Pos1, X_Pos2], -1)

    def word_ent_embedding(self, X, X_Ent1, X_Ent2):
        X_Ent1 = self.word_embedding(X_Ent1).unsqueeze(1).expand(X.shape)
        X_Ent2 = self.word_embedding(X_Ent2).unsqueeze(1).expand(X.shape)
        return torch.cat([X, X_Ent1, X_Ent2], -1)
