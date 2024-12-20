from typing import List, Tuple, Optional, Dict, Any
from transformers.cache_utils import Cache
import torch
import torch.nn.functional as F
import time

def pad_kv(caches):
    """
    List[head, seq, dims]
    """

    shapes = [caches[i].shape[2] for i in range(len(caches))]
    maxlen = max(shapes)
    if all(s == maxlen for s in shapes):
        return torch.concat(caches)

    new_caches = []
    for i in range(len(caches)):
        pad_val = (0, 0, 0, maxlen - caches[i].shape[2], 0, 0, 0, 0)
        pad = F.pad(caches[i], pad_val, value=0.0)
        new_caches.append(pad)
    return torch.concat(new_caches)


class DynamicLengthDecoderCache(Cache):

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.current_uuid = []

    def batch_size(self):
        if len(self.key_cache) > 0:
            return len(self.key_cache[0])
        return 0

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        keys, values = [], []
        for i, k in enumerate(self.current_uuid):
            self.key_cache[layer_idx][k] = torch.cat(
                [self.key_cache[layer_idx][k], key_states[i: i + 1]], dim=-2)
            self.value_cache[layer_idx][k] = torch.cat(
                [self.value_cache[layer_idx][k], value_states[i: i + 1]], dim=-2)
            keys.append(self.key_cache[layer_idx][k])
            values.append(self.value_cache[layer_idx][k])

        k = pad_kv(keys)
        v = pad_kv(values)
        
        return k, v
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        
        lengths = [self.key_cache[0][k].shape[2] for k in self.current_uuid]
        return max(lengths)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

class StaticLengthDecoderCache(Cache):

    def __init__(
        self, 
        batch_size = 20, 
        max_length = 8192,
        device = 'cuda',
        head_size = 16,
        dim_size = 64,
        num_hidden_layers = 24,
        dtype = torch.bfloat16,
        ) -> None:

        self.key_cache, self.value_cache = [], []
        for _ in range(num_hidden_layers):
            self.key_cache.append(
                torch.zeros(
                    batch_size, 
                    head_size, 
                    max_length,
                    dim_size,
                    dtype = torch.bfloat16).to(device)
            )
            self.value_cache.append(
                torch.zeros(
                    batch_size, 
                    head_size, 
                    max_length,
                    dim_size,
                    dtype = torch.bfloat16).to(device)
            )
        self.lengths = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        cache_position = cache_kwargs['cache_position']
        maxlen = max(self.lengths)

        for i in range(len(key_states)):
            self.key_cache[layer_idx][i, :, self.lengths[i]] = key_states[i,:,0]
            self.value_cache[layer_idx][i, :, self.lengths[i]] = value_states[i,:,0]

        k = self.key_cache[layer_idx][:len(key_states), :, :maxlen]
        v = self.value_cache[layer_idx][:len(key_states), :, :maxlen]

        return k, v
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return max(self.lengths)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None