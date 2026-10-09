import numpy as np
import torch
import torch.nn as nn

from graphmixer.models.modules import TimeEncoder
from graphmixer.utils.utils import NeighborSampler


class GraphMixer(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        num_tokens: int,
        num_layers: int = 2,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.1,
        num_channels: int = 64,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(self.device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(self.device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.num_channels = num_channels

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(
                num_tokens=self.num_tokens,
                num_channels=self.num_channels,
                token_dim_expansion_factor=self.token_dim_expansion_factor,
                channel_dim_expansion_factor=self.channel_dim_expansion_factor,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(
            self.num_channels + self.node_feat_dim,
            self.node_feat_dim,
            bias=True,
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
        time_gap: int = 2000,
    ):
        src_node_embeddings = self.compute_node_temporal_embeddings(
            src_node_ids,
            node_interact_times,
            num_neighbors,
            time_gap,
        )
        dst_node_embeddings = self.compute_node_temporal_embeddings(
            dst_node_ids,
            node_interact_times,
            num_neighbors,
            time_gap,
        )

        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
        time_gap: int = 2000,
    ):
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = (
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )
        )

        neighbor_edge_ids_tensor = torch.from_numpy(neighbor_edge_ids).long().to(self.device)
        nodes_edge_raw_features = self.edge_raw_features[neighbor_edge_ids_tensor]

        time_deltas = node_interact_times[:, np.newaxis] - neighbor_times
        nodes_neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(time_deltas).float().to(self.device)
        )
        padding_mask = torch.from_numpy(neighbor_node_ids == 0).to(self.device)
        nodes_neighbor_time_features[padding_mask] = 0.0

        combined_features = torch.cat(
            [nodes_edge_raw_features, nodes_neighbor_time_features],
            dim=-1,
        )
        combined_features = self.projection_layer(combined_features)

        for mlp_mixer in self.mlp_mixers:
            combined_features = mlp_mixer(combined_features)

        combined_features = torch.mean(combined_features, dim=1)

        all_neighbor_ids, owner_ids, neighbor_counts = self.neighbor_sampler.get_neighbors_in_time_window(
            node_ids=node_ids,
            node_interact_times=node_interact_times,
            time_gap=time_gap
        )

        nodes_time_gap_neighbor_node_agg_features = torch.zeros(
            (len(node_ids), self.node_feat_dim),
            device=self.device,
        )
        if len(all_neighbor_ids) > 0:
            all_neighbor_ids = torch.from_numpy(all_neighbor_ids).long().to(self.device)
            owner_ids = torch.from_numpy(owner_ids).long().to(self.device)
            nodes_time_gap_neighbor_node_agg_features.index_add_(
                0,
                owner_ids,
                self.node_raw_features[all_neighbor_ids],
            )
            counts = torch.from_numpy(neighbor_counts).to(self.device).unsqueeze(1)
            nodes_time_gap_neighbor_node_agg_features /= counts.clamp_min(1)

        output_node_features = (
            nodes_time_gap_neighbor_node_agg_features
            + self.node_raw_features[
                torch.from_numpy(node_ids).long().to(self.device)
            ]
        )

        node_embeddings = self.output_layer(
            torch.cat([combined_features, output_node_features], dim=1)
        )

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super().__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        hidden_dim = int(dim_expansion_factor * input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            num_tokens,
            token_dim_expansion_factor,
            dropout,
        )

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            num_channels,
            channel_dim_expansion_factor,
            dropout,
        )

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor

        hidden_tensor = self.channel_norm(output_tensor)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        output_tensor = hidden_tensor + output_tensor

        return output_tensor
