import torch
import torch.nn as nn

class MyPCNN_V(nn.Module):
    def __init__(self, embed_dim, hidden_size=230):
        super(MyPCNN_V, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(embed_dim, hidden_size, 3, padding=1)
        self.hidden_size = hidden_size
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, X_mask):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = torch.tanh(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)