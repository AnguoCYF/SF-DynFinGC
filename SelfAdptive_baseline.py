import warnings
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import dgl
from sklearn.model_selection import train_test_split
from GNN_Classifier import GCNClassifier, GATClassifier, GINClassifier, GraphormerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

#%% Dataset Preprocessing for Aribnb
filepath = '../AirBnb.csv'
data = pd.read_csv(filepath)
data = data.dropna()

features = data.drop(columns=['fraud']).values
labels = data['fraud'].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

class_counts = np.bincount(labels)
class_weights = len(labels) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)


# %% Dataset Preprocessing for South German Credit Card
# filepath = '..\DE_Credit.xlsx'
# data = pd.read_excel(filepath)
# data = data.dropna()
# data['credit_risk'] = data['credit_risk'].replace({1: 0, 0: 1})
#
# data = data.drop(columns=['Id'])
# # 剔除仅包含0的列
# rl_columns = [col for col in data.columns if col.startswith('rl') and data[col].nunique() == 1]
# data_encoded = data.drop(columns=rl_columns)
#
# label_column = 'credit_risk'
# feat_columns = data_encoded.columns.drop([label_column, 'status'])
#
# # 特征标准化
# scaler = StandardScaler()
# data_encoded[feat_columns] = scaler.fit_transform(data_encoded[feat_columns])
#
# # 计算类别权重
# class_counts = np.bincount(data_encoded[label_column].values)
# class_weights = len(data_encoded[label_column].values) / (len(class_counts) * class_counts)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# %%
# 创建图
num_nodes = data_encoded.shape[0]
g = dgl.graph((torch.arange(0, num_nodes), torch.arange(0, num_nodes)))
g = g.to(device)

# 设置节点特征和标签
g.ndata['feat'] = torch.tensor(data_encoded[feat_columns].values, dtype=torch.float32).to(device)
g.ndata['label'] = torch.tensor(data_encoded[label_column].values, dtype=torch.long).to(device)

# 生成训练和测试掩码
train_mask_indices, test_mask_indices = train_test_split(
    np.arange(num_nodes), test_size=0.2, stratify=data_encoded[label_column])
train_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), train_mask_indices)).to(device)
test_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), test_mask_indices)).to(device)
g.train_mask = train_mask
g.test_mask = test_mask


# %%
# 定义自适应邻接矩阵生成器和 GCN 层
class AdaptiveGNN(nn.Module):
    def __init__(self, classifier, nnodes, embed_dim=10, alpha=3.0, topk=20):
        super(AdaptiveGNN, self).__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.topk = topk
        self.nnodes = nnodes

        self.emb1 = nn.Embedding(nnodes, embed_dim)
        self.emb2 = nn.Embedding(nnodes, embed_dim)
        self.lin1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin2 = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.xavier_uniform_(self.emb1.weight)
        nn.init.xavier_uniform_(self.emb2.weight)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

        self.classifier = classifier

    def forward(self, features, idx):
        nodevec1_emb = self.emb1(idx)  # (N, embed_dim), requires_grad=True
        nodevec2_emb = self.emb2(idx)  # (N, embed_dim), requires_grad=True

        # 线性变换 + 激活
        transformed_nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1_emb))  # (N, embed_dim)
        transformed_nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2_emb))  # (N, embed_dim)

        # 计算邻接矩阵（确保所有输入张量启用梯度）
        term1 = torch.mm(transformed_nodevec1, transformed_nodevec2.t())  # (N, N)
        term2 = torch.mm(transformed_nodevec2, transformed_nodevec1.t())  # (N, N)
        a = term1 - term2  # (N, N), requires_grad=True
        adj = F.relu(torch.tanh(self.alpha * a))  # (N, N), requires_grad=True

        mask = torch.zeros_like(adj)
        _, topk_indices = adj.topk(self.topk, dim=1)
        mask.scatter_(1, topk_indices, 1.0)
        adj = adj * mask
        # 添加自环
        adj = adj + torch.eye(self.nnodes).to(adj.device)  # (N, N)

        # 构建图并分类
        src, dst = torch.nonzero(adj > 0, as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=self.nnodes).to(features.device)
        edge_weight = adj[src, dst]
        g.edata['w'] = edge_weight

        logits = self.classifier(g, features)

        return logits, g
# %%
# 训练和评估函数
def train(g, model, epochs, lr):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.train_mask
    test_mask = g.test_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_f1_test = 0
    best_epoch = 0
    best_metrics = (0, 0, 0, 0)

    # 初始化用于累加最后50个epoch性能指标的变量
    sum_loss = 0.0
    sum_acc_train = 0.0
    sum_prec_train = 0.0
    sum_recall_train = 0.0
    sum_f1_train = 0.0
    sum_acc_test = 0.0
    sum_prec_test = 0.0
    sum_recall_test = 0.0
    sum_f1_test = 0.0
    # 确定从哪个epoch开始累加
    start_accumulation = max(0, epochs - 50)

    for epoch in range(epochs):
        model.train()
        logits, constructed_g = model(features, idx=torch.arange(num_nodes).to(device))
        loss = cross_entropy_loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_train, prec_train, recall_train, f1_train = evaluate(model, features, labels, train_mask)
        acc_test, prec_test, recall_test, f1_test = evaluate(model, features, labels, test_mask)
        g = constructed_g

        if f1_test > best_f1_test:
            best_f1_test = f1_test
            best_epoch = epoch
            best_metrics = (acc_test, prec_test, recall_test, f1_test)

        # 从指定的epoch开始累加性能指标
        if epoch >= start_accumulation:
            sum_loss += loss.item()
            sum_acc_train += acc_train
            sum_prec_train += prec_train
            sum_recall_train += recall_train
            sum_f1_train += f1_train
            sum_acc_test += acc_test
            sum_prec_test += prec_test
            sum_recall_test += recall_test
            sum_f1_test += f1_test


        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.3f}, "
                f"Train Acc: {acc_train:.3f}, Test Acc: {acc_test:.3f}, "
                f"Test Prec: {prec_test:.3f}, Test Recall: {recall_test:.3f}, Test F1: {f1_test:.3f}")

    best_acc, best_prec, best_recall, best_f1 = best_metrics
    print(
        f"\nBest Test Performance at Epoch {best_epoch + 1}: "
        f"Acc: {best_acc:.3f}, Prec: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

    # 计算最后50个epoch的平均性能指标
    num_accumulated = epochs - start_accumulation  # 实际累加的epoch数，防止epochs少于50的情况
    avg_loss = sum_loss / num_accumulated
    avg_acc_train = sum_acc_train / num_accumulated
    avg_prec_train = sum_prec_train / num_accumulated
    avg_recall_train = sum_recall_train / num_accumulated
    avg_f1_train = sum_f1_train / num_accumulated
    avg_acc_test = sum_acc_test / num_accumulated
    avg_prec_test = sum_prec_test / num_accumulated
    avg_recall_test = sum_recall_test / num_accumulated
    avg_f1_test = sum_f1_test / num_accumulated

    print("\nAverage Performance over the last 50 epochs:")
    print(f"Train Loss: {avg_loss:.3f}")
    print(f"Train Accuracy: {avg_acc_train:.3f}, Precision: {avg_prec_train:.3f}, Recall: {avg_recall_train:.3f}, F1: {avg_f1_train:.3f}")
    print(f"Test Accuracy: {avg_acc_test:.3f}, Precision: {avg_prec_test:.3f}, Recall: {avg_recall_test:.3f}, F1: {avg_f1_test:.3f}")

    return g

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features, idx=torch.arange(num_nodes).to(device))
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() / len(labels)
        precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    return acc, precision, recall, f1


# %%
# 模型训练
num_classes = len(torch.unique(g.ndata['label']))
in_dim = g.ndata['feat'].shape[1]
cl_hdim = 64
embed_dim = 10
epochs = 300
lr = 0.01

# classifier = GCNClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes).to(device)

classifier = GATClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes, num_heads=8).to(device)

# classifier = GINClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes,  num_layers=2).to(device)

# classifier = GraphormerClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes, num_heads=8, num_layers=2).to(device)

model = AdaptiveGNN(
    classifier=classifier,
    embed_dim=embed_dim,
    nnodes=num_nodes,
    alpha=0.5,  # 控制激活函数饱和度
    topk=15  # 每个节点保留10个邻居
).to(device)

g = train(g, model, epochs, lr)

