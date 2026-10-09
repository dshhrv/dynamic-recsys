import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        super().__init__()
        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    dataset = CustomizedDataset(indices_list=indices_list)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class Data:
    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
    ):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels

        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    graph_df = pd.read_csv(f"./processed_data/{dataset_name}/ml_{dataset_name}.csv")
    edge_raw_features = np.load(f"./processed_data/{dataset_name}/ml_{dataset_name}.npy")
    node_raw_features = np.load(f"./processed_data/{dataset_name}/ml_{dataset_name}_node.npy")

    node_feat_dim = edge_feat_dim = 172

    assert node_feat_dim >= node_raw_features.shape[1], (
        f"Node feature dimension in dataset {dataset_name} is bigger than {node_feat_dim}!"
    )
    assert edge_feat_dim >= edge_raw_features.shape[1], (
        f"Edge feature dimension in dataset {dataset_name} is bigger than {edge_feat_dim}!"
    )

    if node_raw_features.shape[1] < node_feat_dim:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], node_feat_dim - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)

    if edge_raw_features.shape[1] < edge_feat_dim:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], edge_feat_dim - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert (
        node_feat_dim == node_raw_features.shape[1]
        and edge_feat_dim == edge_raw_features.shape[1]
    ), "Unaligned feature dimensions after feature padding!"

    val_time, test_time = np.quantile(
        graph_df.ts,
        [1 - val_ratio - test_ratio, 1 - test_ratio],
    )

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )

    random.seed(2020)

    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    test_node_set = set(src_node_ids[node_interact_times > val_time]) | set(
        dst_node_ids[node_interact_times > val_time]
    )

    new_test_node_set = set(
        random.sample(
            list(test_node_set),
            int(0.1 * num_total_unique_node_ids),
        )
    )

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    observed_edges_mask = np.logical_and(
        ~new_test_source_mask,
        ~new_test_destination_mask,
    )
    train_mask = np.logical_and(
        node_interact_times <= val_time,
        observed_edges_mask,
    )

    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )

    train_node_set = set(train_data.src_node_ids) | set(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0

    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(
        node_interact_times <= test_time,
        node_interact_times > val_time,
    )
    test_mask = node_interact_times > test_time

    edge_contains_new_node_mask = np.array([
        src_node_id in new_node_set or dst_node_id in new_node_set
        for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)
    ])

    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )

    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )

    new_node_val_data = Data(
        src_node_ids=src_node_ids[new_node_val_mask],
        dst_node_ids=dst_node_ids[new_node_val_mask],
        node_interact_times=node_interact_times[new_node_val_mask],
        edge_ids=edge_ids[new_node_val_mask],
        labels=labels[new_node_val_mask],
    )

    new_node_test_data = Data(
        src_node_ids=src_node_ids[new_node_test_mask],
        dst_node_ids=dst_node_ids[new_node_test_mask],
        node_interact_times=node_interact_times[new_node_test_mask],
        edge_ids=edge_ids[new_node_test_mask],
        labels=labels[new_node_test_mask],
    )

    print(
        f"The dataset has {full_data.num_interactions} interactions, "
        f"involving {full_data.num_unique_nodes} different nodes"
    )
    print(
        f"The training dataset has {train_data.num_interactions} interactions, "
        f"involving {train_data.num_unique_nodes} different nodes"
    )
    print(
        f"The validation dataset has {val_data.num_interactions} interactions, "
        f"involving {val_data.num_unique_nodes} different nodes"
    )
    print(
        f"The test dataset has {test_data.num_interactions} interactions, "
        f"involving {test_data.num_unique_nodes} different nodes"
    )
    print(
        f"The new node validation dataset has {new_node_val_data.num_interactions} interactions, "
        f"involving {new_node_val_data.num_unique_nodes} different nodes"
    )
    print(
        f"The new node test dataset has {new_node_test_data.num_interactions} interactions, "
        f"involving {new_node_test_data.num_unique_nodes} different nodes"
    )
    print(
        f"{len(new_test_node_set)} nodes were used for the inductive testing, "
        "i.e. are never seen during training"
    )

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    graph_df = pd.read_csv(f"./processed_data/{dataset_name}/ml_{dataset_name}.csv")
    edge_raw_features = np.load(f"./processed_data/{dataset_name}/ml_{dataset_name}.npy")
    node_raw_features = np.load(f"./processed_data/{dataset_name}/ml_{dataset_name}_node.npy")

    node_feat_dim = edge_feat_dim = 172

    assert node_feat_dim >= node_raw_features.shape[1], (
        f"Node feature dimension in dataset {dataset_name} is bigger than {node_feat_dim}!"
    )
    assert edge_feat_dim >= edge_raw_features.shape[1], (
        f"Edge feature dimension in dataset {dataset_name} is bigger than {edge_feat_dim}!"
    )

    if node_raw_features.shape[1] < node_feat_dim:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], node_feat_dim - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)

    if edge_raw_features.shape[1] < edge_feat_dim:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], edge_feat_dim - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert (
        node_feat_dim == node_raw_features.shape[1]
        and edge_feat_dim == edge_raw_features.shape[1]
    ), "Unaligned feature dimensions after feature padding!"

    val_time, test_time = np.quantile(
        graph_df.ts,
        [1 - val_ratio - test_ratio, 1 - test_ratio],
    )

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(
        node_interact_times <= test_time,
        node_interact_times > val_time,
    )
    test_mask = node_interact_times > test_time

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )

    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )

    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )

    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
    )