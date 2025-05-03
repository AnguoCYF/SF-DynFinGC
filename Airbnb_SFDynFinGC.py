import warnings
import pandas as pd
import os
import numpy as np
from torch import nn
from SF_DynFinGC import *
from GNN_Classifier import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score

from generate_masks import generate_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# %% Preprocessing
data = pd.read_csv('./AirBnb.csv')
features = data.dropna()

label_column = 'fraud'
feat_columns = data.columns.drop(label_column)

scaler = StandardScaler()
features[feat_columns] = scaler.fit_transform(data[feat_columns])

labels = data[label_column]
class_counts = np.bincount(labels)
class_weights = len(labels) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# %%
# 构建图
# initial_g = build_dgl_graph(features.values, labels.values, method='knn', param=10).to(device)

num_nodes = features.shape[0]
initial_g = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes)))
initial_g.ndata['feat'] = torch.tensor(features.values).float()
initial_g.ndata['label'] = torch.tensor(labels.values).long()
initial_g = dgl.add_self_loop(initial_g).to(device)

# 生成掩码
_, _, train_mask, test_mask = generate_masks(features, labels, train_ratio=0.8)
initial_g.train_mask = train_mask.to(device)
initial_g.test_mask = test_mask.to(device)


# %%

def train_model(model, g, lambda_reg=0.2, gamma=2.5, epochs=300, update_epochs=100, lr=1e-3, drop_interval=10, edge_drop_rate=0.2, class_weights=class_weights):

    if class_weights is not None:
        pos_weight = class_weights[1] / class_weights[0]
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        bce_loss_fn = nn.BCEWithLogitsLoss().to(device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    transform = dgl.DropEdge(p=edge_drop_rate)
    best_test_metrics = None
    best_epoch = -1

    # 用于最后50个epoch的性能指标累加
    start_accumulation = max(0, epochs - 50)
    sum_loss = 0.0
    sum_f1_train = 0.0
    sum_acc_test = 0.0
    sum_prec_test = 0.0
    sum_recall_test = 0.0
    sum_f1_test = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        labels_f = labels.float().unsqueeze(1).to(device)  # [N,1]
        h = model(g, features)
        embeds = torch.cat([h, features], dim=1)
        logits = model.classifier(g, embeds)

        task_loss = bce_loss_fn(logits[train_mask], labels_f[train_mask])

        with torch.no_grad():
            if (epoch <= update_epochs):
                adj_rec_sparse = model.update_adjacency_matrix(h.detach(), g, epoch=epoch, total_epochs=update_epochs)
                # 从稀疏邻接矩阵创建新的 DGL 图
                new_g = dgl.graph((adj_rec_sparse.indices()[0], adj_rec_sparse.indices()[1]),
                              num_nodes=adj_rec_sparse.size(0)).to(device)
                new_g.train_mask = train_mask
                new_g.test_mask = test_mask
                new_g.ndata['label'] = labels
            elif (epoch > update_epochs) and (epoch % drop_interval == 0):
                # 应用 DropEdge 引入图结构的微小扰动
                new_g = transform(g)
                adj_rec_sparse = None
            else:
                adj_rec_sparse = None

            g = add_missing_self_loops(new_g)

        if adj_rec_sparse is not None:
            # loss power distribution
            degrees = torch.zeros(adj_rec_sparse.size(0)).to(device)
            degrees.index_add_(0, adj_rec_sparse._indices()[0], adj_rec_sparse._values())
            degrees_cpu = degrees.long().cpu()
            max_degree = degrees_cpu.max().item()
            degree_counts = torch.bincount(degrees_cpu, minlength=max_degree + 1).float().to(device)
            degree_counts = degree_counts[1:]  # 排除度数为零的情况
            P_empirical = degree_counts / degree_counts.sum()

            # 定义目标幂律分布
            k_values = torch.arange(1, degree_counts.size(0) + 1).float().to(device)
            P_k = k_values.pow(-gamma)
            P_k = P_k / P_k.sum()
            loss_degree = torch.nn.functional.mse_loss(P_empirical, P_k)  # 计算度分布损失（使用均方误差）

            loss = task_loss + lambda_reg * loss_degree

        else:
            loss = task_loss

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                probs_train = torch.sigmoid(logits[train_mask])
                probs_test = torch.sigmoid(logits[test_mask])
                predicted_train = (probs_train >= 0.5).long().cpu().numpy()
                predicted_test = (probs_test >= 0.5).long().cpu().numpy()
                train_f1 = f1_score(labels[train_mask].cpu().numpy(), predicted_train, average='macro', zero_division=0)
                test_f1 = f1_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro', zero_division=0)
                test_accuracy = accuracy_score(labels[test_mask].cpu().numpy(), predicted_test)
                test_precision = precision_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro', zero_division=0)
                test_recall = recall_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro', zero_division=0)

                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
                print(f'Train F1: {train_f1:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

                # 更新最佳测试集结果
                if best_test_metrics is None or test_f1 > best_test_metrics[3]:
                    best_test_metrics = (test_accuracy, test_precision, test_recall, test_f1)
                    best_epoch = epoch + 1

        if epoch >= start_accumulation:
            probs_train = torch.sigmoid(logits[train_mask])
            probs_test = torch.sigmoid(logits[test_mask])
            predicted_train = (probs_train >= 0.5).long().cpu().numpy()
            predicted_test = (probs_test >= 0.5).long().cpu().numpy()
            train_f1 = f1_score(labels[train_mask].cpu().numpy(), predicted_train, average='macro', zero_division=0)
            test_f1 = f1_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro', zero_division=0)
            test_accuracy = accuracy_score(labels[test_mask].cpu().numpy(), predicted_test)
            test_precision = precision_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro',
                                             zero_division=0)
            test_recall = recall_score(labels[test_mask].cpu().numpy(), predicted_test, average='macro',
                                       zero_division=0)
            sum_loss += loss.item()
            sum_f1_train += train_f1
            sum_acc_test += test_accuracy
            sum_prec_test += test_precision
            sum_recall_test += test_recall
            sum_f1_test += test_f1

    # 打印最佳测试集结果
    print(f'\nBest Test Performance at Epoch {best_epoch}:')
    print(
        f'Accuracy: {best_test_metrics[0]:.4f}, Precision: {best_test_metrics[1]:.4f}, Recall: {best_test_metrics[2]:.4f}, F1: {best_test_metrics[3]:.4f}')

    num_accumulated = epochs - start_accumulation
    if num_accumulated > 0:
        avg_loss = sum_loss / num_accumulated
        avg_f1_train = sum_f1_train / num_accumulated
        avg_acc_test = sum_acc_test / num_accumulated
        avg_prec_test = sum_prec_test / num_accumulated
        avg_recall_test = sum_recall_test / num_accumulated
        avg_f1_test = sum_f1_test / num_accumulated

        print("\nAverage Performance over the last 50 epochs:")
        print(f"Train Loss: {avg_loss:.3f}")
        print(f"Train F1: {avg_f1_train:.3f}")
        print(f"Test Accuracy: {avg_acc_test:.3f}, Precision: {avg_prec_test:.3f}, "
              f"Recall: {avg_recall_test:.3f}, F1: {avg_f1_test:.3f}")

    return model, g


#%%
num_classes = len(np.unique(labels))
in_dim = features.shape[1]
hidden_dim = 16
cl_dim = 128
epochs = 500
lr = 0.001
dropout = 0.5
k = 20
lambda_reg = 0.5
update_epochs = 200
drop_interval = 50
edge_drop_rate = 0.2
num_heads = 8
num_layers = 2

# 模型初始化
model = DynamicGC(in_feats=in_dim, hidden_feats=hidden_dim, cl_feats=cl_dim, k=k, dropout=dropout, gnn_type='gcn', classifier_type='gcn', num_layers=num_layers, num_heads=num_heads).to(device)

# 调用封装的训练函数
trained_model, g = train_model(model, initial_g, update_epochs=update_epochs, drop_interval=drop_interval, edge_drop_rate=edge_drop_rate,
                            lambda_reg=lambda_reg, epochs=epochs, lr=lr)
