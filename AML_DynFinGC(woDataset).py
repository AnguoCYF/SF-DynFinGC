import warnings
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from generate_masks import generate_masks
from SF_DynFinGC import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
# %% Dataset Preprocessing
# This private dataset, sourced from a Malaysian bank, contains over 30 million transactions spanning
# a two-year period, with an anomaly ratio of $6\%$. Constraint by the contract, can not be provided.

data = pd.read_csv('All_week.csv')
# 移除 'cif_id' 列
data = data.drop(columns=['cif_id', 'account_no'])

categorical_cols = ['account_type', 'product_type', 'business_type']

# 整数编码处理
for col in categorical_cols:
    data[col], _ = pd.factorize(data[col])

# 移除原始的日期列
data_encoded = data.drop(columns=['tran_date_from', 'tran_date_to'])

# 剔除仅包含0的列
rl_columns = [col for col in data_encoded.columns if col.startswith('rl') and data_encoded[col].nunique() == 1]
data_encoded = data_encoded.drop(columns=rl_columns)

# 选择需要标准化的列
columns_to_scale = ['business_type', 'product_type', 'credit_amount', 'credit_count', 'debit_amount', 'debit_count', 'debitbycredit', 'p2', 'debitpluscredit_amount', 'debitpluscredit_count']

# 初始化标准化器
scaler = StandardScaler()

# 对选定列进行标准化
data_encoded[columns_to_scale] = scaler.fit_transform(data_encoded[columns_to_scale])
data_encoded.dropna(inplace=True)

# 计算每个类别的出现次数
class_counts = np.bincount(data_encoded['class'].values)
# 计算每个类别的权重，使用总样本数除以（类别数乘以类别出现次数）
class_weights = len(data_encoded['class'].values) / (len(class_counts) * class_counts)
# 转为Tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 按月份分割数据
data_by_week = {week: data_encoded[data_encoded['week'] == week] for week in data_encoded['week'].unique()}

# 检查每个月份数据的行数，以确保分割正确
data_rows_by_week = {week: subset.shape[0] for week, subset in data_by_week.items()}

#%%

# 初始化用于存储每个月份图的列表和掩码的列表
graphs_by_month = []

# 迭代处理每个月份的数据
for week, data_subset in data_by_week.items():

    features = data_subset.drop(columns=['class', 'week']).values
    labels = data_subset['class'].values

    # 构建图
    # initial_g = build_dgl_graph(features, labels, method='knn', param=1).to(device)

    # 生成一个仅包含自连接的图
    num_nodes = features.shape[0]
    initial_g = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes)))
    initial_g.ndata['feat'] = torch.tensor(features).float()
    initial_g.ndata['label'] = torch.tensor(labels).long()
    initial_g = dgl.add_self_loop(initial_g).to(device)

    # 生成掩码
    _, _, train_mask, test_mask = generate_masks(features, labels, train_ratio=0.8)
    initial_g.train_mask = train_mask.to(device)
    initial_g.test_mask = test_mask.to(device)

    # 存储构建的图和掩码
    graphs_by_month.append(initial_g)

# %%

def train_model(model, dynamic_graphs, lambda_reg=0.2, gamma=2.5, epochs=300, update_epochs=100, lr=1e-3, drop_interval=10, edge_drop_rate=0.2, class_weights=class_weights):
    # 使用 BCEWithLogitsLoss
    if class_weights is not None:
        pos_weight = class_weights[1] / class_weights[0]
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        bce_loss_fn = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1_test = 0
    best_epoch = 0
    best_metrics = (0, 0, 0, 0)
    transform = dgl.DropEdge(p=edge_drop_rate)

    # 预处理图数据
    graph_data = [(g.ndata['feat'], g.ndata['label'], g.train_mask, g.test_mask) for g in dynamic_graphs]

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
        epoch_loss = 0
        all_predicted_train, all_labels_train = [], []
        all_predicted_test, all_labels_test = [], []
        updated_graphs = []

        for i, g in enumerate(dynamic_graphs):
            optimizer.zero_grad()
            features, labels, train_mask, test_mask = graph_data[i]

            # 将标签转换为 float 类型并调整形状为 [N,1]
            labels = labels.float().unsqueeze(1).to(device)  # [N,1]

            h = model(g, features)  # h为节点嵌入表示 [N,hidden_dim]
            # 使用分类器得到 logits，[N,1]
            embeds = torch.cat([h, features], dim=1)
            logits = model.classifier(g, embeds)

            # 计算任务损失
            task_loss = bce_loss_fn(logits[train_mask], labels[train_mask])

            # 图结构更新
            with torch.no_grad():
                if (epoch <= update_epochs):
                    adj_rec_sparse = model.update_adjacency_matrix(h.detach(), g, epoch=epoch, total_epochs=update_epochs)
                    # 从稀疏邻接矩阵创建新的 DGL 图
                    g = dgl.graph((adj_rec_sparse.indices()[0], adj_rec_sparse.indices()[1]),
                                  num_nodes=adj_rec_sparse.size(0)).to(device)
                    g.train_mask = train_mask
                    g.test_mask = test_mask
                    g.ndata['label'] = labels

                elif (epoch > update_epochs) and (epoch % drop_interval == 0):
                    # 应用 DropEdge 引入图结构的微小扰动
                    g = transform(g)
                    adj_rec_sparse = None
                else:
                    adj_rec_sparse = None

                g = add_missing_self_loops(g)
                updated_graphs.append(g)

            if adj_rec_sparse is not None:

                # 计算度分布损失
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

                # 计算度分布损失（均方误差）
                loss_degree = F.mse_loss(P_empirical, P_k)
                loss = task_loss + lambda_reg * loss_degree

            else:
                loss = task_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                probs_train = torch.sigmoid(logits[train_mask])
                probs_test = torch.sigmoid(logits[test_mask])
                predicted_train = (probs_train >= 0.5).long()
                predicted_test = (probs_test >= 0.5).long()
                all_predicted_train.extend(predicted_train.cpu().numpy())
                all_labels_train.extend(labels[train_mask].cpu().numpy())
                all_predicted_test.extend(predicted_test.cpu().numpy())
                all_labels_test.extend(labels[test_mask].cpu().numpy())

        # 更新动态图
        dynamic_graphs = updated_graphs

        # 每10个epoch打印一次性能指标
        if (epoch + 1) % 10 == 0:
            epoch_loss_value = epoch_loss / len(dynamic_graphs)
            f1_train = f1_score(all_labels_train, all_predicted_train, average='macro', zero_division=0)
            acc_test = accuracy_score(all_labels_test, all_predicted_test)
            prec_test = precision_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)
            recall_test = recall_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)
            f1_test = f1_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)
            # print(f"\n Classifier threshold:{model.classifier.cl_thres}")
            print(
                f"Epoch {epoch + 1}/{epochs}, Epoch Loss: {epoch_loss_value:.3f}, Train F1: {f1_train:.3f}, Test Accuracy: {acc_test:.3f}, "
                f"Test Precision: {prec_test:.3f}, Test Recall: {recall_test:.3f}, Test F1: {f1_test:.3f}")

            # 更新最佳性能指标
            if f1_test > best_f1_test:
                best_f1_test = f1_test
                best_epoch = epoch
                best_metrics = (acc_test, prec_test, recall_test, best_f1_test)

        # 累加最后50个epoch的性能指标
        if epoch >= start_accumulation:
            sum_loss += epoch_loss
            sum_f1_train += f1_score(all_labels_train, all_predicted_train, average='macro', zero_division=0)
            sum_acc_test += accuracy_score(all_labels_test, all_predicted_test)
            sum_prec_test += precision_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)
            sum_recall_test += recall_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)
            sum_f1_test += f1_score(all_labels_test, all_predicted_test, average='macro', zero_division=0)

    # 打印最佳性能
    best_acc, best_prec, best_recall, best_f1 = best_metrics
    print(
        f"\nBest Test Performance at Epoch {best_epoch + 1}: "
        f"Accuracy: {best_acc:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

    # 计算最后50个epoch的平均性能指标
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

    return model, dynamic_graphs

#%%
num_classes = len(np.unique(graphs_by_month[0].ndata['label'].cpu().numpy()))
in_dim = features.shape[1]
hidden_dim = 2
cl_dim = 32
epochs = 200
lr = 0.005
dropout = 0.2
k = 2
lambda_reg = 0.2
update_epochs = 50
drop_interval = 100
edge_drop_rate = 0.3
# num_heads = 16
# num_layers = 3

# 模型初始化
model = DynamicGC(in_feats=in_dim, hidden_feats=hidden_dim, cl_feats=cl_dim, k=k, dropout=dropout, gnn_type='gcn', classifier_type='gcn', num_layers=None, num_heads=None).to(device)

# 调用封装的训练函数
trained_model, dynamic_graphs = train_model(model, graphs_by_month, update_epochs=update_epochs, drop_interval=drop_interval , edge_drop_rate=edge_drop_rate,
                            lambda_reg=lambda_reg, epochs=epochs, lr=lr)
