import numpy as np
import torch
from sklearn.model_selection import train_test_split


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