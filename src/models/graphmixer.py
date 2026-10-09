from src.base import BaseModel
from src.utils.clearml_logger import current_run, init_run
import torch
import random
import numpy as np
import pandas as pd
import os

from graphmixer.utils.DataLoader import Data
from graphmixer.utils.utils import get_neighbor_sampler
from graphmixer.models.graphmixer import GraphMixer
from graphmixer.utils.utils import NeighborSampler
from tqdm import tqdm

class GraphMixerModel(BaseModel):
    def __init__(self,
                name="graphmixer",
                time_feat_dim=100,
                num_tokens=20,
                num_layers=2,
                token_dim_expansion_factor=0.5,
                channel_dim_expansion_factor=4.0,
                dropout=0.1,
                learning_rate=1e-4,
                batch_size=200,
                num_epochs=50,
                num_neighbors=10,
                time_gap=2000,
                device="cuda",
                seed=0,
                node_feat_dim=64,
                edge_feat_dim=1,
                num_channels=64,
                eval_item_batch_size=1024,
                num_candidates=128,
                ):
        super().__init__(name)
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_neighbors
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_neighbors = num_neighbors
        self.time_gap = time_gap
        self.seed = seed
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_channels = num_channels
        self.num_candidates = num_candidates
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.n_users = 0
        self.n_items = 0
        self.eval_item_batch_size = eval_item_batch_size
        self.best_epoch = self.num_epochs
        self.best_val_ndcg = -np.inf
        self.best_state_dict = None

    def fit(self, train_dataset, val_dataset=None, metrics=None, max_top_n=20):
        run = current_run()
        created_run = run is None

        if created_run:
            run = init_run(
                project=os.getenv("CLEARML_PROJECT", "recsys-btl"),
                name=os.getenv("CLEARML_TASK_NAME", f"graphmixer-{train_dataset.name}"),
                config={
                    "model": self.name,
                    "dataset": train_dataset.name,
                    "time_feat_dim": self.time_feat_dim,
                    "num_tokens": self.num_tokens,
                    "num_layers": self.num_layers,
                    "token_dim_expansion_factor": self.token_dim_expansion_factor,
                    "channel_dim_expansion_factor": self.channel_dim_expansion_factor,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "num_neighbors": self.num_neighbors,
                    "time_gap": self.time_gap,
                    "num_channels": self.num_channels,
                    "num_candidates": self.num_candidates,
                    "device": str(self.device),
                    "seed": self.seed,
                },
            )
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items

        train_df = (train_dataset._df.sort_values("timestamp", kind="mergesort").reset_index(drop=True))
        self.train_df = train_df.copy()

        if val_dataset is not None:
            self.val_df = val_dataset._df.copy()

        source_nodes = train_df["user_id"].to_numpy(np.int64) + 1
        destination_nodes = (train_df["item_id"].to_numpy(np.int64) + self.n_users + 1)
        timestamps = train_df["timestamp"].to_numpy(np.float64)
        edge_ids = np.arange(1, len(train_df) + 1, dtype=np.int64)

        num_nodes = self.n_users + self.n_items + 1

        node_features = self.rng.normal(size=(num_nodes, self.node_feat_dim)).astype(np.float32)
        edge_features = np.zeros(
            (len(train_df) + 1, self.edge_feat_dim), dtype=np.float32
        )
        edge_features[1:, 0] = 1.0

            
        node_features[0] = 0
        edge_features[0] = 0
        self.train_data = Data(
            src_node_ids=source_nodes,
            dst_node_ids=destination_nodes,
            node_interact_times=timestamps,
            edge_ids=edge_ids,
            labels=np.ones(len(train_df), dtype=np.float32),
        )
        self.train_neighbor_sampler = get_neighbor_sampler(
            data=self.train_data,
            sample_neighbor_strategy="recent",
            seed=self.seed,
        )
        self.model = GraphMixer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=self.train_neighbor_sampler,
            time_feat_dim=self.time_feat_dim,
            num_tokens=self.num_tokens,
            num_layers=self.num_layers,
            token_dim_expansion_factor=self.token_dim_expansion_factor,
            channel_dim_expansion_factor=self.channel_dim_expansion_factor,
            dropout=self.dropout,
            num_channels=self.num_channels,
            device=str(self.device),
        ).to(self.device)
        self.train_edge_raw_features = self.model.edge_raw_features.clone()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        num_interactions = self.train_data.num_interactions
        indices = np.arange(num_interactions)

        for epoch in range(self.num_epochs):
            seen_items_mask = np.zeros((self.n_users + 1, self.n_items), dtype=bool)
            self.model.train()
            self.rng.shuffle(indices)
            epoch_losses = []
            for start in tqdm(range(0, num_interactions, self.batch_size), desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                batch_indices = indices[start:start + self.batch_size]
                users = self.train_data.src_node_ids[batch_indices]
                positive_items = self.train_data.dst_node_ids[batch_indices]
                times = self.train_data.node_interact_times[batch_indices]
                batch_size = len(batch_indices)
                negative_item_ids = self.rng.randint(0, self.n_items, size=(batch_size, self.num_candidates))
                positive_item_ids = positive_items - self.n_users - 1
                collisions = (negative_item_ids == positive_item_ids[:, None]) | seen_items_mask[users[:, None], negative_item_ids]
                while collisions.any():
                    negative_item_ids[collisions] = self.rng.randint(
                        0, self.n_items, size=int(collisions.sum())
                    )
                    collisions = (negative_item_ids == positive_item_ids[:, None]) | seen_items_mask[users[:, None], negative_item_ids]
                seen_items_mask[users, positive_item_ids] = True
                negative_items = (negative_item_ids + self.n_users + 1)
                all_embeddings = self.model.compute_node_temporal_embeddings(
                    node_ids=np.concatenate([users, positive_items, negative_items.reshape(-1)]),
                    node_interact_times=np.concatenate([
                        times,
                        times,
                        np.repeat(times, self.num_candidates),
                    ]),
                    num_neighbors=self.num_neighbors,
                    time_gap=self.time_gap,
                )
                user_embeddings = all_embeddings[:batch_size]
                positive_embeddings = all_embeddings[batch_size:2 * batch_size]
                negative_embeddings = all_embeddings[2 * batch_size:]
                positive_scores = torch.sum(user_embeddings * positive_embeddings, dim=1)
                negative_embeddings = negative_embeddings.view(batch_size, self.num_candidates, -1)
                negative_scores = torch.einsum("bd,bkd->bk", user_embeddings, negative_embeddings)
                loss = -torch.log(torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores) + 1e-8).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            mean_train_loss = float(np.mean(epoch_losses))
            log_dict = {"epoch": epoch + 1, "loss": mean_train_loss}

            if val_dataset is not None and metrics is not None:
                val_predictions = self.predict(val_dataset, top_n=max_top_n, split="validation")
                holdout_users = val_dataset.get_holdout_users()
                val_metrics = metrics(val_predictions[holdout_users, :], val_dataset.get_holdout_array()[holdout_users])
                log_dict.update({f"val/{k}": float(v) for k, v in val_metrics.items()})

                val_ndcg = float(val_metrics["ndcg@20"])
                if val_ndcg > self.best_val_ndcg:
                    self.best_val_ndcg = val_ndcg
                    self.best_epoch = epoch + 1
                    self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

                self.model.set_neighbor_sampler(self.train_neighbor_sampler)
                self.model.edge_raw_features = self.train_edge_raw_features
                self.model.train()

            run.log(log_dict)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, loss={mean_train_loss:.4f}")
        if val_dataset is not None and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if created_run:
            run.finish()
        return self



    @torch.no_grad()
    def predict(self, dataset, top_n, split=None):
        self.model.eval()
        if split is None:
            split = "validation" if dataset._split == "val" else dataset._split
        predictions = np.zeros((dataset.n_users, top_n), dtype=np.int64)
        targets = dataset._holdout_df[["user_id", "timestamp"]]
        if dataset._split == "val":
            history_graph = pd.concat(
                [dataset._df_train, dataset._df_val],
                ignore_index=True,
            )
        else:
            history_graph = pd.concat(
                [dataset._df_train, dataset._df_val, dataset._df_test],
                ignore_index=True,
            )
        history_graph = history_graph.sort_values(
            "timestamp", kind="mergesort"
        ).reset_index(drop=True)
        source_nodes = history_graph["user_id"].to_numpy(np.int64) + 1
        destination_nodes = history_graph["item_id"].to_numpy(np.int64) + self.n_users + 1
        timestamps = history_graph["timestamp"].to_numpy(np.float64)
        edge_ids = np.arange(1, len(history_graph) + 1, dtype=np.int64)
        history_data = Data(src_node_ids=source_nodes, dst_node_ids=destination_nodes, node_interact_times=timestamps, edge_ids=edge_ids, labels=np.ones(len(history_graph), dtype=np.float32))
        eval_neighbor_sampler = get_neighbor_sampler(data=history_data,
                                                    sample_neighbor_strategy="recent",
                                                    seed=self.seed)
        self.model.set_neighbor_sampler(eval_neighbor_sampler)
        history_edge_features = np.zeros((len(history_graph) + 1, self.edge_feat_dim), dtype=np.float32)
        history_edge_features[1:, 0] = 1.0
        self.model.edge_raw_features = torch.from_numpy(history_edge_features).float().to(self.device)
        all_item_ids = np.arange(self.n_items, dtype=np.int64)
        all_item_nodes = all_item_ids + self.n_users + 1

        for row in tqdm(targets.itertuples(index=False), total=len(targets), desc=f"Predict {split}"):
            user_id = int(row.user_id)
            target_time = float(row.timestamp)
            user_node = np.array([user_id + 1], dtype=np.int64)
            user_time = np.array([target_time], dtype=np.float64)

            user_embedding = self.model.compute_node_temporal_embeddings(node_ids=user_node, node_interact_times=user_time, num_neighbors=self.num_neighbors, time_gap=self.time_gap)

            scores = np.empty(self.n_items, dtype=np.float32)
            for start in range(0, self.n_items, self.eval_item_batch_size):
                end = min(start + self.eval_item_batch_size, self.n_items)
                item_nodes = all_item_nodes[start:end]
                item_times = np.full(len(item_nodes), target_time, dtype=np.float64)
                item_embeddings = self.model.compute_node_temporal_embeddings(node_ids=item_nodes, node_interact_times=item_times, num_neighbors=self.num_neighbors, time_gap=self.time_gap)
                scores[start:end] = torch.matmul(item_embeddings, user_embedding.squeeze(0)).cpu().numpy()

            seen_items = history_graph.loc[(history_graph["user_id"] == user_id) & (history_graph["timestamp"] < target_time), "item_id"].to_numpy(np.int64)
            scores[seen_items] = -np.inf
            top_items = np.argpartition(scores, -top_n)[-top_n:]
            top_items = top_items[np.argsort(scores[top_items])[::-1]]
            predictions[user_id] = top_items
        return predictions

    def load_checkpoint(self, path):
        return super().load_checkpoint(path)

    def save_checkpoint(self, path):
        return super().save_checkpoint(path)

    
    def suggest_additional_params(self):
        return {"num_epochs": int(self.best_epoch)}
    
