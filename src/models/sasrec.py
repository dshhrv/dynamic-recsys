from src.base import BaseModel, load_weights, save_weights
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm

from torch.utils.data import DataLoader
from src.metrics import NDCGMetric
from src.utils.clearml_logger import current_run, init_run

def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_seqs(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    if len(user_items) <= maxlen:
        if len(user_items) > 1:
            seq[-len(user_items) + 1:] = user_items[:-1]
        if len(user_items) > 0:
            pos[-len(user_items):] = user_items
    else:
        seq = user_items[-maxlen-1:-1]
        pos = user_items[-maxlen:]

    return seq, pos, neg
    
def collate_fn_sasrec(x, maxlen, pad_token, is_train):

    seqs_batch = []
    pos_batch = []
    user_ids = []
    seen_history = []

    for user in x:
        seq, pos, neg = pad_seqs(user["history"].tolist(), maxlen, pad_token)

        seqs_batch.append(seq)
        pos_batch.append(pos)
        user_ids.append(user["user_id"])
        if not is_train:
            seen_history.append(user["history"])

    batch = {
        "seq": torch.LongTensor(np.asarray(seqs_batch)),
        "pos": torch.LongTensor(np.asarray(pos_batch)),
        "user_id": torch.LongTensor(np.asarray(user_ids)),
        "seen_history": seen_history
    }

    return batch


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.ff = nn.Sequential(
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.ff(
            inputs.transpose(-1,-2)
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRecBackBone(nn.Module):
    def __init__(
            self,
            item_num,
            hidden_units,
            dropout_rate,
            maxlen,
            num_blocks,
            num_heads,
            manual_seed=37
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                hidden_units,num_heads,dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(manual_seed)
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :, :] * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def score(self, seq):

        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecModel(BaseModel):
    def __init__(
        self,
        name: str = "sasrec",
        hidden_size: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-3,
        device: str = "cuda",
        seq_len: int = 50,
        n_epochs: int = 2000,
        batch_size: int = 64,
        seed: int = 52,
        log_step: int = 200,
        mix_x: bool = True,
        alpha: float = 2.0,
        beta: float = 1.0,
        bucket_size_y: int = 256,
        patience_per_epoch: int = 1,
        max_patience: int = 20,
        val_top_n: int = 20,
        filter_seen: bool = True
    ):
        BaseModel.__init__(self, name)

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.filter_seen = filter_seen

        self.alpha = alpha
        self.beta = beta

        self.patience_per_epoch = patience_per_epoch
        self.max_patience = max_patience

        self.mix_x = mix_x

        self.bucket_size_y = bucket_size_y

        self.log_step = log_step
        self.val_top_n = val_top_n
    
    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            manual_seed=37
        )

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def _sasrec_forward(self, log_seqs, pos_seqs):
        emb = self.model.log2feats(log_seqs)

        x = emb.reshape(-1, emb.shape[-1])
        y = pos_seqs.reshape(-1).long()

        mask = y != self.model.pad_token
        x = x[mask]
        y = y[mask]

        logits = x @ self.model.item_emb.weight.T
        logits[:, self.model.pad_token] = float("-inf")

        return F.cross_entropy(logits, y)

    def suggest_additional_params(self):
        return {"n_epochs": self.n_epochs}
            
    
    def fit(
        self,
        train_dataset,
        val_dataset,
        metrics=None,
        max_top_n=20,
        checkpoint_path=None,
        resume_epoch=0,
    ):
        """
        Fit the model to the dataset.
        """

        self._post_init(
            train_dataset,
            val_dataset
        )
        if resume_epoch and checkpoint_path is not None:
            load_weights(self.model, checkpoint_path)

        run = current_run()
        if run is None:
            run = init_run(
                project=os.getenv("CLEARML_PROJECT", "recsys-btl"),
                name=os.getenv("CLEARML_TASK_NAME", f"sasrec-{train_dataset.name}"),
                config={
                    "model": self.name,
                    "hidden_size": self.hidden_size,
                    "num_blocks": self.num_blocks,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                    "lr": self.lr,
                    "seq_len": self.seq_len,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "filter_seen": self.filter_seen,
                },
            )

        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle = True,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, True)
        )

        best_metric = -1
        patience_cnt = 0
        best_epoch = -1
        best_state_dict = None

        for i in range(resume_epoch, self.n_epochs):
            self.model.train()

            train_loss = []

            index = 0

            for batch in train_dataloader:

                loss = self._sasrec_forward(
                    batch["seq"].to(self.device),
                    batch["pos"].to(self.device),
                )

                self.opt.zero_grad()

                train_loss.append(loss.item())

                loss.backward()
                
                self.opt.step() 

                index += 1

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")    
            
            
            mean_train_loss = sum(train_loss) / len(train_loss)
            print(f"Mean train loss: {mean_train_loss}")

            epoch_logs = {
                "epoch": i + 1,
                "train/loss": mean_train_loss,
            }

            stop_training = False
            if (i + 1) % self.patience_per_epoch == 0 and val_dataset is not None:

                holdout_users = val_dataset.get_holdout_users()
        
                predictions = self.predict(
                    val_dataset,
                    max(max_top_n, self.val_top_n)
                )

                val_predictions = predictions[holdout_users, :]
                val_targets = val_dataset.get_holdout_array()[holdout_users]
                if metrics is None:
                    metric = NDCGMetric(self.val_top_n)
                    val_metrics = {
                        metric.name: metric(val_predictions, val_targets)
                    }
                else:
                    val_metrics = metrics(val_predictions, val_targets)

                epoch_logs.update({
                    f"val/{name}": float(value)
                    for name, value in val_metrics.items()
                })
                metric_name = f"ndcg@{self.val_top_n}"
                metric_val = float(val_metrics[metric_name])
                print(f"Value of {metric_name}: {metric_val}")

                if best_metric < metric_val:
                    best_metric = metric_val
                    patience_cnt = 0

                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }

                    best_epoch = i + 1
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.max_patience:
                        stop_training = True

            run.log(epoch_logs, step=i + 1)
            if checkpoint_path is not None:
                save_weights(self.model, checkpoint_path)
            if stop_training:
                break

        if best_metric >= 0:
            self.model.load_state_dict(best_state_dict)
            self.n_epochs = best_epoch

            run.summary["best_val_metric"] = float(best_metric)
            run.summary["best_epoch"] = int(best_epoch)

        run.finish()





                
    @torch.no_grad()
    def predict(self, dataset, top_n: int, batch_size: int = 128):
        self.model.eval()

        n_users = dataset.n_users
        
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle = False,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, False)
        )
    
        recommendations = np.zeros((n_users, top_n))

        for batch in dataloader:

            logits = self.model.score(
                seq=batch["pos"].to(self.device),
            )

            if self.filter_seen:
                for i in range(len(batch["seen_history"])):
                    if batch["seen_history"][i].shape[0] > 0:
                        logits[i][batch["seen_history"][i]] = -10 ** 9
    
            top_items = torch.topk(
                logits,
                k=top_n,
                dim=1
            ).indices
    
            for uid, recs in zip(batch["user_id"], top_items):
                recommendations[uid, :] = recs.cpu().numpy()
                
        return recommendations
    
