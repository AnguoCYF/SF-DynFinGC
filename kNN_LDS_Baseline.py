import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 复用已封装的下游GNN分类器
from GNN_Classifier import GCNClassifier, GATClassifier, GINClassifier, GraphormerClassifier

def generate_masks(features, labels, train_ratio, valid_ratio=0.2, seed=42, resample=False):
    if resample:
        sm = SMOTE(random_state=seed)
        features, labels = sm.fit_resample(features, labels)

    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 第一次分割：训练集 + 测试集
    _, _, _, _, train_idx, test_idx = train_test_split(
        features, labels, np.arange(num_nodes),
        train_size=train_ratio,
        random_state=seed
    )

    # 第二次分割：从训练集中划分出验证集
    train_sub_idx, valid_idx = train_test_split(
        train_idx,
        train_size=1 - valid_ratio,  # 训练集占原始训练集的比例
        random_state=seed
    )

    # 生成掩码
    train_mask[train_sub_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True

    return features, labels, train_mask, valid_mask, test_mask

class LearnableGraphStructure(nn.Module):
    """可学习的图结构生成模块"""

    def __init__(self, num_nodes, init_method='knn', k=5, features=None):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # kNN初始化
        if init_method == 'knn' and features is not None:
            with torch.no_grad():
                knn_adj = kneighbors_graph(features, k, metric='cosine').toarray()
                self.theta.data = torch.tensor(knn_adj, dtype=torch.float32)

        # 上三角稀疏约束
        mask = torch.triu(torch.ones_like(self.theta), diagonal=1)
        self.register_buffer('mask', mask)

    def sample_adj(self, training=True):
        prob = torch.sigmoid(self.theta) * self.mask

        if training:
            # Gumbel-Softmax风格重参数化
            noise = torch.rand_like(prob)
            sample = (noise < prob).float()
            adj = sample + (prob - prob.detach())  # 直通估计器
        else:
            adj = (prob > 0.5).float() + (prob - prob.detach())  # 直通估计器

        return adj

class LDS_Trainer:
    """LDS-GNN训练框架（严格遵循论文算法）"""

    def __init__(self, features, labels, train_mask, val_mask, test_mask, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化图结构学习器
        self.graph_learner = LearnableGraphStructure(
            features.shape[0],
            init_method=config['graph_init'],
            k=config['k'],
            features=features
        ).to(self.device)

        num_classes = len(np.unique(labels))
        # 根据config选择图分类器（复用GNN_Classifier.py中的实现，并增加Graphormer支持）
        if config['model_type'] == 'GCN':
            self.model = GCNClassifier(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                num_classes=num_classes,
                dropout=config.get('dropout', 0.5)
            ).to(self.device)
        elif config['model_type'] == 'GIN':
            self.model = GINClassifier(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                num_classes=num_classes,
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.5)
            ).to(self.device)
        elif config['model_type'] == 'GAT':
            self.model = GATClassifier(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                num_classes=num_classes,
                num_heads=config.get('num_heads', 4),
                dropout=config.get('dropout', 0.5)
            ).to(self.device)
        elif config['model_type'] == 'Graphormer':
            self.model = GraphormerClassifier(
                in_dim=features.shape[1],
                hidden_dim=config['hidden_dim'],
                num_classes=num_classes,
                num_heads=config.get('num_heads', 4),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.5)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")

        # 优化器（分离参数）
        self.optim_model = torch.optim.Adam(
            self.model.parameters(),
            lr=config['inner_lr'],
            weight_decay=config['l2_reg']
        )
        self.optim_graph = torch.optim.Adam(
            [self.graph_learner.theta],
            lr=config['outer_lr']
        )

        # 数据准备
        self.features = torch.FloatTensor(features).to(self.device)
        self.labels = torch.LongTensor(labels).to(self.device)
        self.train_mask = torch.as_tensor(train_mask).to(self.device)
        self.val_mask = torch.as_tensor(val_mask).to(self.device)
        self.test_mask = torch.as_tensor(test_mask).to(self.device)

    def train_step(self):
        """内层优化：固定θ优化模型参数"""
        self.model.train()

        # 采样邻接矩阵
        adj = self.graph_learner.sample_adj(training=False)

        # 构建DGL图
        rows, cols = torch.where(adj > 0)
        edge_weights = adj[rows, cols]
        g = dgl.graph((rows, cols), num_nodes=adj.shape[0]).to(self.device)
        g.edata['w'] = edge_weights
        g = g.to(self.device)

        # 前向传播
        logits = self.model(g, self.features)
        loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])

        # 反向传播
        self.optim_model.zero_grad()
        loss.backward()
        self.optim_model.step()
        return loss.item()

    def update_graph_structure(self, val_loss):
        """外层优化：固定模型优化θ"""
        self.optim_graph.zero_grad()
        val_loss.backward()

        # 梯度投影（仅保留上三角部分）
        with torch.no_grad():
            if self.graph_learner.theta.grad is None:
                raise RuntimeError("theta.grad 未计算！检查计算图连接性。")
            grad = self.graph_learner.theta.grad * self.graph_learner.mask
            self.graph_learner.theta.data -= self.config['outer_lr'] * grad
            # print(self.graph_learner.theta.grad)

        return val_loss.item()

    def evaluate(self, mask):
        """评估函数 (支持任意掩码)"""
        self.model.eval()
        with torch.set_grad_enabled(True):  # 评估时不追踪梯度
            adj = self.graph_learner.sample_adj(training=True)
            rows, cols = torch.where(adj > 0)  # 获取非零边的行列索引
            edge_weights = adj[rows, cols]  # 边的概率值
            g = dgl.graph((rows, cols), num_nodes=adj.shape[0]).to(self.device)
            g.edata['w'] = edge_weights

            logits = self.model(g, self.features)
            loss = F.cross_entropy(logits[mask], self.labels[mask])

            preds = logits[mask].argmax(1).cpu().numpy()
            labels = self.labels[mask].cpu().numpy()

            acc = (preds == labels).mean()
            prec = precision_score(labels, preds, average='macro', zero_division=0)
            recall = recall_score(labels, preds, average='macro', zero_division=0)
            f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return loss, acc, prec, recall, f1
    def get_final_graph(self):
        """构建最终图结构"""
        with torch.no_grad():
            adj = self.graph_learner.sample_adj(training=False)
            rows, cols = torch.where(adj > 0)
            edge_weights = adj[rows, cols]
            g = dgl.graph((rows, cols), num_nodes=adj.shape[0]).to(self.device)
            g.edata['w'] = edge_weights
            g.ndata['feat'] = self.features  # 添加节点特征
            g.ndata['label'] = self.labels    # 添加标签
            self.final_graph = g

        return self.final_graph

    def train(self):
        """双层优化训练循环"""
        best_val_acc = 0
        patience = 0

        for epoch in range(self.config['max_epochs']):
            # 内层优化（多次参数更新）
            for _ in range(self.config['inner_steps']):
                train_loss = self.train_step()

            # 外层优化（基于验证损失）
            val_loss, val_acc, _, _, _ = self.evaluate(self.val_mask)
            _, test_acc, test_prec, test_recall, test_f1 = self.evaluate(self.test_mask)

            self.update_graph_structure(val_loss)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Acc: {test_acc:.3f}\n"
                  f"Test Metrics: Prec {test_prec:.3f}, Recall {test_recall:.3f}, F1 {test_f1:.3f}")

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_test_metrics = (test_acc, test_prec, test_recall, test_f1)
            else:
                patience += 1
                if patience >= self.config['patience']:
                    break

        print(f"\nBest Test Metrics: "
              f"Acc: {best_test_metrics[0]:.3f}, "
              f"Prec: {best_test_metrics[1]:.3f}, "
              f"Recall: {best_test_metrics[2]:.3f}, "
              f"F1: {best_test_metrics[3]:.3f}")

        return self

if __name__ == "__main__":
    # %% Data Format for AML Dataset
    data = pd.read_csv('D:\OneDrive_Personal\OneDrive\GNN_Exp\DynFinGC\All_month.csv')
    # 移除 'cif_id' 列
    data = data.drop(columns=['cif_id', 'account_no'])
    data.dropna(inplace=True)

    categorical_cols = ['account_type', 'product_type', 'business_type']
    # 整数编码处理
    for col in categorical_cols:
        data[col], _ = pd.factorize(data[col])

    # 移除原始的日期列
    data_encoded = data.drop(columns=['tran_date_from', 'tran_date_to'])

    # 剔除仅包含0的列
    rl_columns = [col for col in data_encoded.columns if col.startswith('rl') and data_encoded[col].nunique() == 1]
    data_encoded = data_encoded.drop(columns=rl_columns)

    features = data_encoded.drop(columns=['class'])
    labels = data_encoded['class'].values

    # 选择需要标准化的列
    columns_to_scale = ['business_type', 'product_type', 'credit_amount', 'credit_count', 'debit_amount', 'debit_count',
                        'debitbycredit', 'p2', 'debitpluscredit_amount', 'debitpluscredit_count']

    # 初始化标准化器
    scaler = StandardScaler()
    # 对选定列进行标准化
    features[columns_to_scale] = scaler.fit_transform(features[columns_to_scale])
    features = features.values

    # %% Data format for Airbnb Dataset
    # data = pd.read_csv('/home/lab_comp/tmp/Pycharm_DynFinGC/AirBnb.csv')
    # data = data.dropna()
    #
    # features = data.drop(columns=['fraud'])
    # labels = data['fraud']
    #
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # %% Data format for DE Dataset
    # file_path = '/home/lab_comp/tmp/Pycharm_DynFinGC/DE_Credit.xlsx'
    # data = pd.read_excel(file_path)
    # data = data.dropna()
    # data['credit_risk'] = data['credit_risk'].replace({1: 0, 0: 1})
    #
    # data = data.drop(columns=['Id'])
    # # 剔除仅包含0的列
    # rl_columns = [col for col in data.columns if col.startswith('rl') and data[col].nunique() == 1]
    # data_encoded = data.drop(columns=rl_columns)
    #
    # label_column = 'credit_risk'
    # feat_columns = data_encoded.columns.drop([label_column])
    #
    # scaler = StandardScaler()
    # data_encoded[feat_columns] = scaler.fit_transform(data_encoded[feat_columns])
    #
    # features = data_encoded[feat_columns].values
    # labels = data_encoded[label_column].values

#%% 生成掩码
    features, labels, train_mask, val_mask, test_mask = generate_masks(features, labels, train_ratio=0.8, seed=42)

    # 配置参数（与论文实验设置一致）
    config = {
        'model_type': 'GCN',  # 可选: 'GCN', 'GIN', 'GAT', 'Graphormer'
        'hidden_dim': 128,
        'inner_lr': 0.01,
        'outer_lr': 0.1,
        'l2_reg': 5e-4,
        'inner_steps': 10,
        'max_epochs': 500,
        'patience': 30,
        'graph_init': 'knn',
        'k': 10,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.5
    }

    # 训练LDS-GNN
    trainer = LDS_Trainer(features, labels, train_mask, val_mask, test_mask, config)
    trainer = trainer.train()
    final_g = trainer.get_final_graph()

