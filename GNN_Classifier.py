import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv, GINConv, GraphormerLayer

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.5):
        super(GCNClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出维度设为1

        # 定义可学习的阈值参数
        self.cl_thres = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = self.conv2(g, h)
        h = self.dropout(h)
        logits_raw = self.fc(h)  # [N,1]
        # Shift logits by cl_thres
        shifted_logit = logits_raw - self.cl_thres  # 可学习的阈值
        return shifted_logit  # 返回偏移后的logit

class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.5):
        super(GATClassifier, self).__init__()
        out_feats = hidden_dim // num_heads
        self.gat1 = GATConv(in_dim, out_feats, num_heads=num_heads, feat_drop=dropout, activation=F.relu)
        self.gat2 = GATConv(out_feats * num_heads, out_feats * num_heads, num_heads=1, feat_drop=dropout, activation=None)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_feats * num_heads, 1)  # 输出维度设为1

        # 定义可学习的阈值参数
        self.cl_thres = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, g, features):
        h = self.gat1(g, features)  # [N, hidden_dim]
        h = h.flatten(1)  # [N, hidden_dim]
        h = self.gat2(g, h)  # [N, hidden_dim, 1]
        h = h.mean(1)  # [N, hidden_dim]
        h = self.dropout(h)
        logits_raw = self.fc(h)  # [N,1]
        # Shift logits by cl_thres
        shifted_logit = logits_raw - self.cl_thres  # 可学习的阈值
        return shifted_logit  # 返回偏移后的logit

class GINClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.5):
        super(GINClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # 输入层
        self.layers.append(GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ), 'sum'))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ), 'sum'))

        self.fc = nn.Linear(hidden_dim, 1)  # 输出维度设为1

        # 定义可学习的阈值参数
        self.cl_thres = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, g, features):
        h = features
        for conv in self.layers:
            h = conv(g, h)
            h = self.dropout(h)
        logits_raw = self.fc(h)  # [N,1]
        # Shift logits by cl_thres
        shifted_logit = logits_raw - self.cl_thres  # 可学习的阈值
        return shifted_logit  # 返回偏移后的logit

class GraphormerClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.5):
        super(GraphormerClassifier, self).__init__()
        if hidden_dim % num_heads != 0:
            hidden_dim = hidden_dim - (hidden_dim % num_heads)
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            GraphormerLayer(
                feat_size=hidden_dim,
                hidden_size=hidden_dim * num_heads,
                num_heads=num_heads,
                attn_bias_type='add',
                norm_first=False,
                dropout=dropout,
                attn_dropout=dropout,
                activation=nn.ReLU()
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出维度设为1

        # 定义可学习的阈值参数
        self.cl_thres = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, g, features):
        # features: (N, in_dim)
        batch_size = 1
        N = features.size(0)
        h = self.embedding(features)  # h: (N, hidden_dim)
        h = h.unsqueeze(0)  # h: (1, N, hidden_dim)

        # 创建 attn_bias: (1, N, N, num_heads)
        attn_bias = torch.zeros((batch_size, N, N, self.num_heads), device=h.device)

        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias)

        h = h.squeeze(0)  # h: (N, hidden_dim)
        h = self.dropout(h)
        logits_raw = self.fc(h)  # [N,1]
        # Shift logits by cl_thres
        shifted_logit = logits_raw - self.cl_thres  # 可学习的阈值
        return shifted_logit  # 返回偏移后的logit
