import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv, GINConv

from GNN_Classifier import GCNClassifier, GATClassifier, GINClassifier, GraphormerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import dgl
import torch


def add_missing_self_loops(g):
    """
    为缺少自环的节点添加自环，避免重复添加。

    :param g: 当前的 DGLGraph
    :return: 添加了必要自环的 DGLGraph
    """
    num_nodes = g.num_nodes()
    all_nodes = torch.arange(num_nodes, device=g.device)

    # 获取现有的自环边
    u, v = g.edges()
    self_loops_mask = u == v
    nodes_with_self_loop = torch.unique(u[self_loops_mask])

    # 找出缺少自环的节点
    mask = torch.ones(num_nodes, dtype=torch.bool, device=g.device)
    mask[nodes_with_self_loop] = False
    nodes_missing_self_loop = all_nodes[mask]

    if nodes_missing_self_loop.numel() > 0:
        # 创建自环边
        new_self_loops = torch.stack([nodes_missing_self_loop, nodes_missing_self_loop], dim=0)
        # 添加自环边
        g = dgl.add_edges(g, new_self_loops[0], new_self_loops[1])
        # print(f"添加了 {nodes_missing_self_loop.numel()} 个自环。")

    return g


class DynamicGC(nn.Module):
    def __init__(self, in_feats, hidden_feats, cl_feats, dropout=0.5, k=10, gnn_type='gcn', classifier_type='gcn', num_heads=4, num_layers=2):
        super(DynamicGC, self).__init__()

        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.k = k
        self.beta = nn.Parameter(torch.tensor(1.0, device=device))
        if self.gnn_type == 'gcn':
            self.conv1 = GraphConv(in_feats, hidden_feats)
            self.conv2 = GraphConv(hidden_feats, hidden_feats)
        elif self.gnn_type == 'sage':
            self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
            self.conv2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        elif self.gnn_type == 'gat':
            self.conv1 = GATConv(in_feats, hidden_feats // num_heads, num_heads=num_heads)
            self.conv2 = GATConv(hidden_feats, hidden_feats, num_heads=1)
        elif self.gnn_type == 'gin':
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ), aggregator_type='sum')
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ), aggregator_type='sum')
        else:
            raise ValueError("Unsupported GNN type. Choose from 'gcn', 'sage', 'gat', or 'gin'.")

        # 根据 classifier_type 来选择下游分类器
        classifier_type = classifier_type.lower()
        if classifier_type == 'gcn':
            self.classifier = GCNClassifier(in_dim=in_feats + hidden_feats, hidden_dim=cl_feats, dropout_rate=dropout).to(device)
        elif classifier_type == 'gat':
            self.classifier = GATClassifier(in_dim=in_feats + hidden_feats, hidden_dim=cl_feats, num_heads=num_heads, dropout=dropout).to(device)
        elif classifier_type == 'gin':
            self.classifier = GINClassifier(in_dim=in_feats + hidden_feats, hidden_dim=cl_feats, num_layers=num_layers,dropout=dropout).to(device)
        elif classifier_type == 'graphormer':
            self.classifier = GraphormerClassifier(in_dim=in_feats + hidden_feats, hidden_dim=cl_feats, num_heads=num_heads, num_layers=num_layers , dropout=dropout).to(device)
        elif classifier_type == 'linear':
            self.classifier = nn.Linear(in_feats + hidden_feats, 1).to(device)
        else:
            raise ValueError("Unsupported classifier type. Choose from 'gcn', 'gat', 'gin', or 'graphormer'.")

        self.adj_rec_sparse = None
        self.layer_norm = nn.LayerNorm(hidden_feats)

    def forward(self, g, x):
        # 使用 GCN 对图进行特征提取
        h = self.conv1(g, x)
        if self.gnn_type == 'gat':
            h = h.view(h.size(0), -1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        if self.gnn_type == 'gat':
            h = h.squeeze(1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer_norm(h)

        return h

    #%% By threshold
    def update_adjacency_matrix_bythres(self, h):
        # 计算相似度矩阵
        adj_rec = torch.mm(self.beta * h, h.t())
        adj_rec = torch.sigmoid(adj_rec)
        # 去除对角线元素（自环）
        adj_rec = adj_rec - torch.diag_embed(torch.diag(adj_rec))

        # 确定阈值，例如使用平均值加标准差
        mean = adj_rec.mean()
        std = adj_rec.std()
        threshold = mean + 0.5 * std
        threshold = min(threshold, 0.99)
        print(f'Threshold: {threshold:.3f}')
        # 保留相似度大于阈值的边
        adj_rec = (adj_rec >= threshold).float() * adj_rec

        # 转换为稀疏矩阵
        adj_rec_sparse = adj_rec.to_sparse()
        return adj_rec_sparse

#%%
    def update_adjacency_matrix(self, h, g, epoch, total_epochs, final_k=1,
                                edge_limit_ratio=0.70, replacement_ratio=0.1):
        """
        Update adjacency matrix by combining both strategies:
        1. Decay k strategy: Dynamically adjust k based on the current epoch.
        2. Edge limit strategy: Ensure the total number of edges does not exceed a certain ratio of maximum possible edges.

        :param h: Node embeddings [N, hidden_dim]
        :param g: Current DGLGraph
        :param epoch: Current training epoch (starting from 0)
        :param total_epochs: Total number of epochs
        :param final_k: The minimum value k should decay to by the end of training
        :param edge_limit_ratio: Ratio of the maximum possible edges allowed in the graph.
        :param replacement_ratio: Fraction of existing edges to remove when the limit is exceeded
        :return: Sparse adjacency matrix
        """
        # 1. Decay k strategy
        initial_k = self.k
        dynamic_k = int(initial_k * (1 - epoch / total_epochs) + final_k * (epoch / total_epochs))
        dynamic_k = max(dynamic_k, final_k)  # Ensure it doesn't go below final_k
        k = dynamic_k

        h = F.normalize(h, p=2, dim=1)
        num_nodes = h.size(0)

        # 2. Compute edge limit
        # Maximum possible edges for an undirected graph without self-loops:
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        edge_limit = int(edge_limit_ratio * max_possible_edges)

        # 3. Compute similarity matrix and remove self-loops
        adj_rec = torch.mm(self.beta * h, h.t())
        adj_rec = torch.sigmoid(adj_rec)
        adj_rec = adj_rec - torch.diag_embed(torch.diag(adj_rec))

        # 4. Get existing edges and find unique undirected edges
        existing_row, existing_col = g.edges()
        min_indices = torch.min(existing_row, existing_col)
        max_indices = torch.max(existing_row, existing_col)
        unique_existing_edges = torch.stack([min_indices, max_indices], dim=0)
        unique_existing_edges = torch.unique(unique_existing_edges, dim=1)
        num_unique_existing_edges = unique_existing_edges.size(1)

        # 5. If adding k new edges exceeds the edge limit, remove some edges first
        if num_unique_existing_edges + k > edge_limit:
            num_edges_to_remove = (num_unique_existing_edges + k) - edge_limit
            num_edges_to_remove = min(num_edges_to_remove, int(num_unique_existing_edges * replacement_ratio))

            if num_edges_to_remove > 0:
                # Randomly remove some existing edges
                remove_indices = torch.randperm(num_unique_existing_edges, device=h.device)[:num_edges_to_remove]
                edges_to_keep = torch.ones(num_unique_existing_edges, dtype=torch.bool, device=h.device)
                edges_to_keep[remove_indices] = False
                unique_existing_edges = unique_existing_edges[:, edges_to_keep]

        # 6. After removal (if any), extract the updated existing edges
        new_existing_row = unique_existing_edges[0]
        new_existing_col = unique_existing_edges[1]

        # 7. Create a mask to exclude existing edges from new edge selection
        mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=h.device)
        mask[new_existing_row, new_existing_col] = False
        mask[new_existing_col, new_existing_row] = False

        # 8. To avoid double counting edges (u,v) and (v,u), consider only upper triangular part
        adj_rec_masked = adj_rec.masked_fill(~mask, float('-inf')).triu(diagonal=1)

        # 9. Select Top-k edges
        top_k_values, top_k_indices = torch.topk(adj_rec_masked.view(-1), k)

        # 10. Convert 1D indices to 2D
        row_indices_new = top_k_indices // num_nodes
        col_indices_new = top_k_indices % num_nodes

        valid_mask = top_k_values != float('-inf')
        row_indices_new = row_indices_new[valid_mask]
        col_indices_new = col_indices_new[valid_mask]

        # 11. Add reverse edges to build an undirected graph
        row_indices_rev = col_indices_new
        col_indices_rev = row_indices_new

        # 12. Combine new edges and their reverse
        row_indices_new = torch.cat([row_indices_new, row_indices_rev], dim=0)
        col_indices_new = torch.cat([col_indices_new, col_indices_rev], dim=0)
        values_new = torch.ones_like(row_indices_new, dtype=torch.float, device=h.device)

        # 13. Combine old edges and new edges
        # Existing edges are currently in unique form, add them with both directions:
        row_indices_combined = torch.cat([new_existing_row.repeat_interleave(2), row_indices_new], dim=0)
        col_indices_combined = torch.cat([new_existing_col.repeat_interleave(2), col_indices_new], dim=0)
        values_combined = torch.cat([
            torch.ones(new_existing_row.size(0)*2, dtype=torch.float, device=h.device),
            values_new
        ], dim=0)

        # 14. Create the final coalesced sparse adjacency matrix
        adj_rec_sparse = torch.sparse_coo_tensor(
            torch.stack([row_indices_combined, col_indices_combined], dim=0),
            values_combined,
            (num_nodes, num_nodes),
            device=h.device
        ).coalesce()

        return adj_rec_sparse
