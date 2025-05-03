import warnings
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import dgl
from dgl.nn import GraphConv
from sklearn.model_selection import train_test_split
from GNN_Classifier import *
from Trad_GC import *

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


#%%
num_nodes = features.shape[0]

# 创建自连接图
g = dgl.graph((torch.arange(0, num_nodes), torch.arange(0, num_nodes)))
g = dgl.add_self_loop(g).to(device)

# 设置节点特征和标签
g.ndata['feat'] = torch.tensor(features, dtype=torch.float32).to(device)
g.ndata['label'] = torch.tensor(labels, dtype=torch.long).to(device)

# 生成训练和测试掩码
train_mask_indices, test_mask_indices = train_test_split(np.arange(num_nodes), test_size=0.2, stratify=labels, random_state=42)
train_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), train_mask_indices)).to(device)
test_mask = torch.BoolTensor(np.isin(np.arange(num_nodes), test_mask_indices)).to(device)
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)


# %%
# 定义 VGAE 模型
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv_mu = GraphConv(hidden_dim, latent_dim)
        self.conv_logstd = GraphConv(hidden_dim, latent_dim)

    def forward(self, g, x):
        h = self.conv1(g, x)
        mu = self.conv_mu(g, h)
        logstd = self.conv_logstd(g, h)
        return mu, logstd

class InnerProductDecoder(nn.Module):
    def forward(self, z):
        # 使用内积解码器重构邻接矩阵
        adj_reconstructed = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_reconstructed

class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(VGAEModel, self).__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = InnerProductDecoder()

    def forward(self, g, x):
        mu, logvar = self.encoder(g, x)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decoder(z)
        return z, mu, logvar, adj_reconstructed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# %%
def train(g, vgae_model, classifier, epochs, lr, threshold=0.8):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.train_mask
    test_mask = g.test_mask

    optimizer = torch.optim.Adam(list(vgae_model.parameters()) + list(classifier.parameters()), lr=lr)
    cross_entropy_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

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
        vgae_model.train()
        classifier.train()
        z, mu, logvar, adj_reconstructed = vgae_model(g, features)

        # 构建重构的图
        adj_reconstructed = adj_reconstructed.detach()
        adj_reconstructed = adj_reconstructed.cpu()
        adj_reconstructed.fill_diagonal_(0)  # 移除自环
        # 根据阈值二值化
        adj_binary = (adj_reconstructed > threshold).float()
        # 获取边的索引
        src, dst = torch.nonzero(adj_binary, as_tuple=True)
        # 创建新的图
        reconstructed_g = dgl.graph((src, dst), num_nodes=num_nodes).to(device)
        reconstructed_g = dgl.add_self_loop(reconstructed_g)
        reconstructed_g.ndata['feat'] = features

        # 前向传播
        logits = classifier(reconstructed_g, reconstructed_g.ndata['feat'])

        # 分类损失
        loss_cls = cross_entropy_loss_fn(logits[train_mask], labels[train_mask])

        # KL 散度损失
        kl_loss = 0.5 * torch.mean(torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1, dim=1))

        # 总损失
        loss = loss_cls + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train, prec_train, recall_train, f1_train = evaluate(labels, logits, train_mask)
        acc_test, prec_test, recall_test, f1_test = evaluate(labels, logits, test_mask)

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
                f"Train Accuracy: {acc_train:.3f}, Test Accuracy: {acc_test:.3f}, "
                f"Test Precision: {prec_test:.3f}, Test Recall: {recall_test:.3f}, Test F1: {f1_test:.3f}")

    best_acc, best_prec, best_recall, best_f1 = best_metrics
    print(
        f"\nBest Test Performance at Epoch {best_epoch + 1}: "
        f"Accuracy: {best_acc:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

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


def evaluate(labels, logits, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    acc = correct.item() * 1.0 / len(labels)
    precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    return acc, precision, recall, f1

# %%
num_classes = len(torch.unique(g.ndata['label']))
in_dim = g.ndata['feat'].shape[1]
gnn_dim = 128
latent_dim = 64
epochs = 500
lr = 0.005
cl_hdim = 64

vgae_model = VGAEModel(in_dim, hidden_dim=gnn_dim, latent_dim=latent_dim).to(device)
classifier = GraphormerClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes).to(device)

train(g, vgae_model, classifier, epochs, lr, threshold=0.8)
