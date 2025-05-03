import warnings
import pandas as pd
from torch import nn
from Trad_GC import *
from GNN_Classifier import *
from sklearn.preprocessing import StandardScaler
import numpy as np

import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score

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
_, _, train_mask, test_mask = generate_masks(features, labels, train_ratio=0.8)

# 构建图
static_graph = build_dgl_graph(features, labels, method='knn', param=10).to(device)

# 创建自连接图
num_nodes = features.shape[0]
# static_graph = dgl.graph((torch.arange(0, num_nodes), torch.arange(0, num_nodes)))
# static_graph = dgl.add_self_loop(static_graph).to(device)
# static_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32).to(device)
# static_graph.ndata['label'] = torch.tensor(labels, dtype=torch.long).to(device)

static_graph.train_mask = train_mask.to(device)
static_graph.test_mask = test_mask.to(device)

# %%
def train(static_graph, model, epochs, lr):
    labels = static_graph.ndata['label']
    train_mask = static_graph.train_mask
    test_mask = static_graph.test_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    best_f1_test = 0
    best_epoch = 0
    best_metrics = (0, 0, 0, 0)
    g = static_graph.to(device)

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
        logits = model(g, g.ndata['feat'])
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train, prec_train, recall_train, f1_train = evaluate(g, model, train_mask)
        acc_test, prec_test, recall_test, f1_test = evaluate(g, model, test_mask)

        # 更新最佳性能指标
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
                f"Test Precision: {prec_test:.3f}, Test Recall: {recall_test:.3f}, Test F1: {f1_test:.3f}"
            )

    best_acc, best_prec, best_recall, best_f1 = best_metrics
    print(
        f"\nBest Test Performance at Epoch {best_epoch + 1}: "
        f"Accuracy: {best_acc:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}"
    )

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

def evaluate(g, model, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        logits = logits[mask]
        labels = g.ndata['label'][mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    return acc, precision, recall, f1

# %%
num_classes = len(np.unique(labels))
in_dim = static_graph.ndata['feat'].shape[1]
cl_hdim = 128
gnn_dim = 128
epochs = 500
lr = 0.005
dropout = 0.3

classifier = GCNClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes).to(device)

# classifier = GCNClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes).to(device)

# classifier = GATClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes, num_heads=4).to(device)

# classifier = GINClassifier(in_dim=in_dim, hidden_dim=cl_hdim, num_classes=num_classes,  num_layers=2).to(device)

train(static_graph, classifier, epochs, lr)
