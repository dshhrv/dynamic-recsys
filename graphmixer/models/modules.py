import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super().__init__()

        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        frequencies = 1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
        self.w.weight = nn.Parameter(torch.from_numpy(frequencies).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        timestamps = timestamps.unsqueeze(dim=2)
        return torch.cos(self.w(timestamps))


class MergeLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, (
            "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"
        )

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(
            self.query_dim,
            num_heads * self.head_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            self.key_dim,
            num_heads * self.head_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.key_dim,
            num_heads * self.head_dim,
            bias=False,
        )

        self.scaling_factor = self.head_dim ** -0.5
        self.layer_norm = nn.LayerNorm(self.query_dim)
        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
    ):
        node_features = node_features.unsqueeze(dim=1)

        query = residual = torch.cat([node_features, node_time_features], dim=2)
        query = self.query_projection(query).reshape(
            query.shape[0],
            query.shape[1],
            self.num_heads,
            self.head_dim,
        )

        key = value = torch.cat(
            [
                neighbor_node_features,
                neighbor_node_edge_features,
                neighbor_node_time_features,
            ],
            dim=2,
        )

        key = self.key_projection(key).reshape(
            key.shape[0],
            key.shape[1],
            self.num_heads,
            self.head_dim,
        )
        value = self.value_projection(value).reshape(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.head_dim,
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = torch.einsum("bhld,bhnd->bhln", query, key)
        attention = attention * self.scaling_factor

        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        attention = attention.masked_fill(attention_mask, -1e10)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        attention_output = torch.einsum("bhln,bhnd->bhld", attention_scores, value)
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        output = self.dropout(self.residual_fc(attention_output))
        output = self.layer_norm(output + residual)
        output = output.squeeze(dim=1)

        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(attention_dim, 4 * attention_dim),
            nn.Linear(4 * attention_dim, attention_dim),
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim),
        ])

    def forward(
        self,
        inputs_query: torch.Tensor,
        inputs_key: torch.Tensor = None,
        inputs_value: torch.Tensor = None,
        neighbor_masks: np.ndarray = None,
    ):
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query

        transposed_inputs_query = inputs_query.transpose(0, 1)
        transposed_inputs_key = inputs_key.transpose(0, 1)
        transposed_inputs_value = inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        hidden_states = self.multi_head_attention(
            query=transposed_inputs_query,
            key=transposed_inputs_key,
            value=transposed_inputs_value,
            key_padding_mask=neighbor_masks,
        )[0].transpose(0, 1)

        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        hidden_states = self.linear_layers[1](
            self.dropout(F.relu(self.linear_layers[0](outputs)))
        )
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs