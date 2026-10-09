import random

import numpy as np
import torch
import torch.nn as nn

from graphmixer.utils.DataLoader import Data


def set_random_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)

    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]

    return res


def get_parameter_sizes(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


class NeighborSampler:
    def __init__(
        self,
        adj_list: list,
        sample_neighbor_strategy: str = "uniform",
        time_scaling_factor: float = 0.0,
        seed: int = None,
    ):
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == "time_interval_aware":
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        for node_idx, per_node_neighbors in enumerate(adj_list):
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])

            self.nodes_neighbor_ids.append(
                np.array([x[0] for x in sorted_per_node_neighbors])
            )
            self.nodes_edge_ids.append(
                np.array([x[1] for x in sorted_per_node_neighbors])
            )
            self.nodes_neighbor_times.append(
                np.array([x[2] for x in sorted_per_node_neighbors])
            )

            if self.sample_neighbor_strategy == "time_interval_aware":
                neighbor_times = np.array(
                    [x[2] for x in sorted_per_node_neighbors]
                )
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(neighbor_times)
                )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_neighbors_in_time_window(self, node_ids: np.ndarray, node_interact_times: np.ndarray, time_gap: float):
        neighbor_chunks = []
        neighbor_counts = np.zeros(len(node_ids), dtype=np.int64)

        for owner_id, (node_id, interact_time) in enumerate(zip(node_ids, node_interact_times)):
            times = self.nodes_neighbor_times[node_id]

            left = np.searchsorted(
                times,
                interact_time - time_gap,
                side="right"
            )
            right = np.searchsorted(
                times,
                interact_time,
                side="left"
            )
            neighbor_ids = self.nodes_neighbor_ids[node_id][left:right]
            neighbor_ids = np.unique(neighbor_ids)
            neighbor_counts[owner_id] = len(neighbor_ids)
            if len(neighbor_ids) > 0:
                neighbor_chunks.append(neighbor_ids)

        if neighbor_chunks:
            all_neighbor_ids = np.concatenate(neighbor_chunks).astype(np.int64, copy=False)
            owner_ids = np.repeat(np.arange(len(node_ids), dtype=np.int64), neighbor_counts)
        else:
            all_neighbor_ids = np.empty(0, dtype=np.int64)
            owner_ids = np.empty(0, dtype=np.int64)

        return all_neighbor_ids, owner_ids, neighbor_counts
    
    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        if len(node_neighbor_times) == 0:
            return np.array([])

        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        exp_node_neighbor_times = np.exp(
            self.time_scaling_factor * node_neighbor_times
        )
        sampled_probabilities = (
            exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        )
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10

        return sampled_probabilities

    def find_neighbors_before(
        self,
        node_id: int,
        interact_time: float,
        return_sampled_probabilities: bool = False,
    ):
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                self.nodes_neighbor_sampled_probabilities[node_id][:i],
            )

        return (
            self.nodes_neighbor_ids[node_id][:i],
            self.nodes_edge_ids[node_id][:i],
            self.nodes_neighbor_times[node_id][:i],
            None,
        )

    def get_historical_neighbors(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
        assert num_neighbors > 0, (
            "Number of sampled neighbors for each node should be greater than 0!"
        )

        nodes_neighbor_ids = np.zeros(
            (len(node_ids), num_neighbors)
        ).astype(np.longlong)
        nodes_edge_ids = np.zeros(
            (len(node_ids), num_neighbors)
        ).astype(np.longlong)
        nodes_neighbor_times = np.zeros(
            (len(node_ids), num_neighbors)
        ).astype(np.float32)

        for idx, (node_id, node_interact_time) in enumerate(
            zip(node_ids, node_interact_times)
        ):
            (
                node_neighbor_ids,
                node_edge_ids,
                node_neighbor_times,
                node_neighbor_sampled_probabilities,
            ) = self.find_neighbors_before(
                node_id=node_id,
                interact_time=node_interact_time,
                return_sampled_probabilities=(
                    self.sample_neighbor_strategy == "time_interval_aware"
                ),
            )

            if len(node_neighbor_ids) == 0:
                continue

            if self.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
                if node_neighbor_sampled_probabilities is not None:
                    node_neighbor_sampled_probabilities = torch.softmax(
                        torch.from_numpy(
                            node_neighbor_sampled_probabilities
                        ).float(),
                        dim=0,
                    ).numpy()

                if self.seed is None:
                    sampled_indices = np.random.choice(
                        a=len(node_neighbor_ids),
                        size=num_neighbors,
                        p=node_neighbor_sampled_probabilities,
                    )
                else:
                    sampled_indices = self.random_state.choice(
                        a=len(node_neighbor_ids),
                        size=num_neighbors,
                        p=node_neighbor_sampled_probabilities,
                    )

                nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                sorted_position = nodes_neighbor_times[idx, :].argsort()
                nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]

            elif self.sample_neighbor_strategy == "recent":
                node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                node_edge_ids = node_edge_ids[-num_neighbors:]
                node_neighbor_times = node_neighbor_times[-num_neighbors:]

                start = num_neighbors - len(node_neighbor_ids)
                nodes_neighbor_ids[idx, start:] = node_neighbor_ids
                nodes_edge_ids[idx, start:] = node_edge_ids
                nodes_neighbor_times[idx, start:] = node_neighbor_times

            else:
                raise ValueError(
                    f"Not implemented error for sample_neighbor_strategy "
                    f"{self.sample_neighbor_strategy}!"
                )

        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(
        self,
        num_hops: int,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
        assert num_hops > 0, "Number of sampled hops should be greater than 0!"

        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
            self.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )
        )

        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]

        for hop in range(1, num_hops):
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
                self.get_historical_neighbors(
                    node_ids=nodes_neighbor_ids_list[-1].flatten(),
                    node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                    num_neighbors=num_neighbors,
                )
            )

            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        return (
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
        )

    def get_all_first_hop_neighbors(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ):
        nodes_neighbor_ids_list = []
        nodes_edge_ids_list = []
        nodes_neighbor_times_list = []

        for node_id, node_interact_time in zip(node_ids, node_interact_times):
            (
                node_neighbor_ids,
                node_edge_ids,
                node_neighbor_times,
                _,
            ) = self.find_neighbors_before(
                node_id=node_id,
                interact_time=node_interact_time,
                return_sampled_probabilities=False,
            )

            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return (
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
        )

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(
    data: Data,
    sample_neighbor_strategy: str = "uniform",
    time_scaling_factor: float = 0.0,
    seed: int = None,
):
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    adj_list = [[] for _ in range(max_node_id + 1)]

    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
        data.src_node_ids,
        data.dst_node_ids,
        data.edge_ids,
        data.node_interact_times,
    ):
        adj_list[src_node_id].append(
            (dst_node_id, edge_id, node_interact_time)
        )
        adj_list[dst_node_id].append(
            (src_node_id, edge_id, node_interact_time)
        )

    return NeighborSampler(
        adj_list=adj_list,
        sample_neighbor_strategy=sample_neighbor_strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
    )


class NegativeEdgeSampler:
    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        interact_times: np.ndarray = None,
        last_observed_time: float = None,
        negative_sample_strategy: str = "random",
        seed: int = None,
    ):
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times

        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != "random":
            self.possible_edges = {
                (src_node_id, dst_node_id)
                for src_node_id in self.unique_src_node_ids
                for dst_node_id in self.unique_dst_node_ids
            }

        if self.negative_sample_strategy == "inductive":
            self.observed_edges = self.get_unique_edges_between_start_end_time(
                self.earliest_time,
                self.last_observed_time,
            )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(
        self,
        start_time: float,
        end_time: float,
    ):
        selected_time_interval = np.logical_and(
            self.interact_times >= start_time,
            self.interact_times <= end_time,
        )

        return {
            (src_node_id, dst_node_id)
            for src_node_id, dst_node_id in zip(
                self.src_node_ids[selected_time_interval],
                self.dst_node_ids[selected_time_interval],
            )
        }

    def sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray = None,
        batch_dst_node_ids: np.ndarray = None,
        current_batch_start_time: float = 0.0,
        current_batch_end_time: float = 0.0,
    ):
        if self.negative_sample_strategy == "random":
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(
                size=size
            )

        elif self.negative_sample_strategy == "historical":
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )

        elif self.negative_sample_strategy == "inductive":
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(
                size=size,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time,
            )

        else:
            raise ValueError(
                f"Not implemented error for negative_sample_strategy "
                f"{self.negative_sample_strategy}!"
            )

        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(
                0,
                len(self.unique_src_node_ids),
                size,
            )
            random_sample_edge_dst_node_indices = np.random.randint(
                0,
                len(self.unique_dst_node_ids),
                size,
            )
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(
                0,
                len(self.unique_src_node_ids),
                size,
            )
            random_sample_edge_dst_node_indices = self.random_state.randint(
                0,
                len(self.unique_dst_node_ids),
                size,
            )

        return (
            self.unique_src_node_ids[random_sample_edge_src_node_indices],
            self.unique_dst_node_ids[random_sample_edge_dst_node_indices],
        )

    def random_sample_with_collision_check(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
    ):
        assert batch_src_node_ids is not None
        assert batch_dst_node_ids is not None

        batch_edges = {
            (batch_src_node_id, batch_dst_node_id)
            for batch_src_node_id, batch_dst_node_id in zip(
                batch_src_node_ids,
                batch_dst_node_ids,
            )
        }

        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0

        random_edge_indices = self.random_state.choice(
            len(possible_random_edges),
            size=size,
            replace=len(possible_random_edges) < size,
        )

        negative_src_node_ids = np.array([
            possible_random_edges[idx][0]
            for idx in random_edge_indices
        ])
        negative_dst_node_ids = np.array([
            possible_random_edges[idx][1]
            for idx in random_edge_indices
        ])

        return negative_src_node_ids, negative_dst_node_ids

    def historical_sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ):
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time,
            end_time=current_batch_start_time,
        )
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time,
            end_time=current_batch_end_time,
        )

        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array(
            [edge[0] for edge in unique_historical_edges]
        )
        unique_historical_edges_dst_node_ids = np.array(
            [edge[1] for edge in unique_historical_edges]
        )

        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)

            random_sample_src_node_ids, random_sample_dst_node_ids = (
                self.random_sample_with_collision_check(
                    size=num_random_sample_edges,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                )
            )

            negative_src_node_ids = np.concatenate([
                random_sample_src_node_ids,
                unique_historical_edges_src_node_ids,
            ])
            negative_dst_node_ids = np.concatenate([
                random_sample_dst_node_ids,
                unique_historical_edges_dst_node_ids,
            ])

        else:
            historical_sample_edge_node_indices = self.random_state.choice(
                len(unique_historical_edges),
                size=size,
                replace=False,
            )
            negative_src_node_ids = unique_historical_edges_src_node_ids[
                historical_sample_edge_node_indices
            ]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[
                historical_sample_edge_node_indices
            ]

        return (
            negative_src_node_ids.astype(np.longlong),
            negative_dst_node_ids.astype(np.longlong),
        )

    def inductive_sample(
        self,
        size: int,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        current_batch_start_time: float,
        current_batch_end_time: float,
    ):
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time,
            end_time=current_batch_start_time,
        )
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time,
            end_time=current_batch_end_time,
        )

        unique_inductive_edges = (
            historical_edges
            - self.observed_edges
            - current_batch_edges
        )
        unique_inductive_edges_src_node_ids = np.array(
            [edge[0] for edge in unique_inductive_edges]
        )
        unique_inductive_edges_dst_node_ids = np.array(
            [edge[1] for edge in unique_inductive_edges]
        )

        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)

            random_sample_src_node_ids, random_sample_dst_node_ids = (
                self.random_sample_with_collision_check(
                    size=num_random_sample_edges,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                )
            )

            negative_src_node_ids = np.concatenate([
                random_sample_src_node_ids,
                unique_inductive_edges_src_node_ids,
            ])
            negative_dst_node_ids = np.concatenate([
                random_sample_dst_node_ids,
                unique_inductive_edges_dst_node_ids,
            ])

        else:
            inductive_sample_edge_node_indices = self.random_state.choice(
                len(unique_inductive_edges),
                size=size,
                replace=False,
            )
            negative_src_node_ids = unique_inductive_edges_src_node_ids[
                inductive_sample_edge_node_indices
            ]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[
                inductive_sample_edge_node_indices
            ]

        return (
            negative_src_node_ids.astype(np.longlong),
            negative_dst_node_ids.astype(np.longlong),
        )

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)
