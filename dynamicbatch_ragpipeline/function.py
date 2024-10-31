import torch
import torch.nn.functional as F
import logging

def efficient_attention_mask(batch_size, max_len, lengths, device, dtype, ones=True):
    lengths = torch.tensor(lengths)
    left = torch.arange(max_len).expand(
        batch_size, 1, 1, max_len)
    right = lengths.view(
        batch_size, 1, 1, 1)
    if ones:
        mask = left < right
        mask = mask.float()
    else:
        mask = left > right
        mask = mask.float().masked_fill_(mask, torch.finfo(dtype).min)
    return mask.to(device).type(dtype)

def cleanup_cache(cache):
    try:
        if isinstance(cache, tuple) or isinstance(cache, list):
            cache = list(cache)
            for i in range(len(cache)):
                cache[i] = list(cache[i])
                for _ in range(len(cache[i])):
                    del cache[i][0]

        else:
            for _ in range(len(cache.key_cache)):
                del cache.key_cache[0]
            for _ in range(len(cache.value_cache)):
                del cache.value_cache[0]
    except Exception as e:
        logging.warning('failed to clear cache')
