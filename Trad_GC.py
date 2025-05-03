import dgl
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree

def normalize_matrix(matrix):
    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    normalized_matrix = np.nan_to_num(normalized_matrix)
    return normalized_matrix


def generate_mahdistance_threshold_adjacency_matrix(features, quantile=10):

    # 计算马氏距离矩阵
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(features)
    inv_cov = np.linalg.inv(cov_estimator.covariance_)
    mahalanobis_distances = pdist(features, metric='mahalanobis', VI=inv_cov)
    distance_matrix = squareform(mahalanobis_distances)
    distance_matrix = normalize_matrix(distance_matrix)  # 归一化马氏距离矩阵

    threshold = np.percentile(distance_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (distance_matrix <= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix

def generate_euclidean_threshold_adjacency_matrix(features, quantile=10):

    # 计算欧氏距离矩阵
    euclidean_distances = pdist(features, metric='euclidean')
    distance_matrix = squareform(euclidean_distances)
    distance_matrix = normalize_matrix(distance_matrix)  # 归一化欧氏距离矩阵

    threshold = np.percentile(distance_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (distance_matrix <= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def generate_cosine_similarity_matrix(features, quantile=90):
    similarity_matrix = cosine_similarity(features)

    threshold = np.percentile(similarity_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (similarity_matrix >= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def generate_knn_adjacency_matrix(features, k=10):
    # 获取样本数量
    n_samples = features.shape[0]

    # 处理样本数量为1的情况
    if n_samples == 1:
        return np.ones((1, 1))  # 返回一个1x1的零矩阵

    # 确保 k 不大于样本数量
    if k >= n_samples:
        k = n_samples - 1  # 如果 k 超过样本数量，设置为 n_samples - 1

    # 生成稀疏的邻接矩阵
    adjacency_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)  # 确保是无向图

    return adjacency_matrix

def generate_mst_adjacency_matrix(features):
    # 计算欧氏距离矩阵
    euclidean_distances = pdist(features, metric='euclidean')
    distance_matrix = squareform(euclidean_distances)

    # 构建最小生成树
    mst = minimum_spanning_tree(distance_matrix)

    # 将最小生成树转换为邻接矩阵
    adjacency_matrix = mst.toarray()

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def build_dgl_graph(features, labels, method, param, fuse_knn=10, fuse_euc=5, fuse_mah=5, fuse_cos=95):

    if method == 'cosine':
        adjacency_matrix = generate_cosine_similarity_matrix(features, param)
    elif method == 'knn':
        adjacency_matrix = generate_knn_adjacency_matrix(features, param)
    elif method == 'mahalanobis':
        adjacency_matrix = generate_mahdistance_threshold_adjacency_matrix(features, param)
    elif method == 'euclidean':
        adjacency_matrix = generate_euclidean_threshold_adjacency_matrix(features, param)
    elif method == 'mst':
        adjacency_matrix = generate_mst_adjacency_matrix(features)
    else:
        raise ValueError("Invalid method. Choose from 'mst', 'euclidean', 'cosine', 'knn', or 'mahalanobis'.")

    if not isinstance(adjacency_matrix, csr_matrix):
        adjacency_matrix = csr_matrix(adjacency_matrix)


    # 将稀疏矩阵转换为COO格式以便提取边信息
    adjacency_matrix_coo = adjacency_matrix.tocoo()
    src_nodes = adjacency_matrix_coo.row
    dst_nodes = adjacency_matrix_coo.col

    dgl_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=features.shape[0])

    edge_weights = torch.ones(dgl_graph.number_of_edges(), dtype=torch.float32)

    # 添加自环，此步骤会增加新的边
    dgl_graph = dgl.add_self_loop(dgl_graph)
    num_self_loops = dgl_graph.number_of_nodes()
    self_loop_weights = torch.ones(num_self_loops, dtype=torch.float32)
    edge_weights = torch.cat([edge_weights, self_loop_weights], dim=0)

    dgl_graph.edata['w'] = edge_weights.view(-1, 1)
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
    dgl_graph.ndata['label'] = torch.tensor(labels, dtype=torch.long)

    print(f'the {method}--{param} graph construction complete.\n')

    return dgl_graph

def generate_masks(features, labels, train_ratio, seed=42, resample=False):

    if resample == True:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=seed)
        features, labels = sm.fit_resample(features, labels)

    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 使用train_test_split进行数据集划分
    _, _, _, _, train_idx, test_idx = train_test_split(features, labels, np.arange(num_nodes), train_size=train_ratio, random_state=seed)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return features, labels, train_mask, test_mask







