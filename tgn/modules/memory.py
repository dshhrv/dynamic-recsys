import torch
from torch import nn
from collections import defaultdict


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension,
                 message_dimension=None, device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.combination_method = combination_method
        self.register_buffer(
            "memory",
            torch.zeros(
                (self.n_nodes, self.memory_dimension), device=self.device
            ),
        )
        self.register_buffer(
            "last_update",
            torch.zeros(self.n_nodes, device=self.device),
        )
        self.__init_memory__()

    def __init_memory__(self):
        with torch.no_grad():
            self.memory.zero_()
            self.last_update.zero_()
        self.messages = defaultdict(list)

    def _node_indices(self, node_idxs):
        return torch.as_tensor(
            node_idxs, dtype=torch.long, device=self.memory.device
        )

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        node_idxs = self._node_indices(node_idxs)
        return self.memory[node_idxs, :]

    def get_all_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        node_idxs = self._node_indices(node_idxs)
        with torch.no_grad():
            self.memory[node_idxs, :] = values.detach()

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return (
            self.memory.detach().clone(),
            self.last_update.detach().clone(),
            messages_clone
        )

    def restore_memory(self, memory_backup):
        with torch.no_grad():
            self.memory.copy_(memory_backup[0])
            self.last_update.copy_(memory_backup[1])

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))
            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []
