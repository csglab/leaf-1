"""Window pooling for fragment-level embeddings.

Splits each fragment into [CLS] [MOD] left_flank [CUT] body [CUT] right_flank and
pools left/body/right into a fixed number of windows. Output is the concatenation
of the pooled windows plus the four special-token embeddings, producing the
9,984-dim per-fragment representation when used with the 768-dim encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiWindowPooler(nn.Module):
    """Pools variable-length sequence embeddings into a fixed number of windows."""

    def __init__(self, num_windows=3):
        super().__init__()
        self.num_windows = num_windows
        self.pool = nn.AdaptiveAvgPool1d(num_windows)

    def forward(self, embeddings, mask=None):
        if isinstance(embeddings, torch.Tensor) and mask is not None:
            batch_size, seq_len, embed_dim = embeddings.shape
            device = embeddings.device

            if not mask.any():
                return torch.zeros(batch_size, self.num_windows * embed_dim, device=device)

            masked_embeddings = embeddings * mask.unsqueeze(-1)

            valid_proportions = F.adaptive_avg_pool1d(
                mask.float().unsqueeze(1), self.num_windows
            ).squeeze(1)
            valid_proportions = valid_proportions.clamp(min=1e-6)

            x_avg = F.adaptive_avg_pool1d(
                masked_embeddings.permute(0, 2, 1),
                self.num_windows,
            )
            x = x_avg / valid_proportions.unsqueeze(1)
            x = x.permute(0, 2, 1)
            return x.reshape(batch_size, -1)

        elif isinstance(embeddings, torch.Tensor):
            batch_size, seq_len, embed_dim = embeddings.shape
            device = embeddings.device

            if seq_len == 0:
                return torch.zeros(batch_size, self.num_windows * embed_dim, device=device)

            x = embeddings.permute(0, 2, 1)
            x = self.pool(x)
            x = x.permute(0, 2, 1)
            return x.reshape(batch_size, -1)


def extract_regions(embeddings, start_pos, end_pos):
    """Boolean masks for left flank, fragment body, and right flank.

    Left flank starts after [CLS] and [MOD] (pos >= 2). start_pos and end_pos
    are the indices of the first and second [CUT] tokens.
    """
    batch_size, seq_len, _ = embeddings.shape
    device = embeddings.device

    if (start_pos + 1 >= end_pos).any():
        invalid_idx = (start_pos + 1 >= end_pos).nonzero().squeeze()
        raise ValueError(f"Start/end positions invalid for batch rows: {invalid_idx.tolist()}")

    pos = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

    left_mask = (pos >= 2) & (pos < start_pos.unsqueeze(1))
    frag_mask = (pos > start_pos.unsqueeze(1)) & (pos < end_pos.unsqueeze(1))
    right_mask = pos > end_pos.unsqueeze(1)

    return left_mask, frag_mask, right_mask


def compute_window_embeddings(embeddings, start_pos, end_pos, num_windows=3):
    """Concatenate left/frag/right window pools with CUT1, CUT2, CLS, MOD tokens."""
    batch_size = embeddings.size(0)
    device = embeddings.device

    left_mask, frag_mask, right_mask = extract_regions(embeddings, start_pos, end_pos)

    pooler = AdaptiveMultiWindowPooler(num_windows)
    left_pooled = pooler(embeddings, left_mask)
    frag_pooled = pooler(embeddings, frag_mask)
    right_pooled = pooler(embeddings, right_mask)

    batch_indices = torch.arange(batch_size, device=device)
    cut1_tokens = embeddings[batch_indices, start_pos]
    cut2_tokens = embeddings[batch_indices, end_pos]
    cls_tokens = embeddings[:, 0, :]
    mod_tokens = embeddings[:, 1, :]

    return torch.cat(
        [left_pooled, frag_pooled, right_pooled, cut1_tokens, cut2_tokens, cls_tokens, mod_tokens],
        dim=1,
    )


def compute_window_embeddings_from_counts(embeddings, n_token_left, n_token_frag, n_token_right,
                                          num_windows=3):
    """Same layout as compute_window_embeddings for inputs without [CUT] tokens (nocut).
    Regions come from the token counts; the two CUT chunks are zeros."""
    batch_size, seq_len, embed_dim = embeddings.shape
    device = embeddings.device
    pos = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
    lo = 2 + n_token_left.unsqueeze(1)
    hi = lo + n_token_frag.unsqueeze(1)
    end = hi + n_token_right.unsqueeze(1)
    if (n_token_frag <= 0).any():
        raise ValueError("every row needs at least one body token")

    pooler = AdaptiveMultiWindowPooler(num_windows)
    left_pooled = pooler(embeddings, (pos >= 2) & (pos < lo))
    frag_pooled = pooler(embeddings, (pos >= lo) & (pos < hi))
    right_pooled = pooler(embeddings, (pos >= hi) & (pos < end))
    zeros = torch.zeros(batch_size, embed_dim, device=device, dtype=embeddings.dtype)

    return torch.cat(
        [left_pooled, frag_pooled, right_pooled, zeros, zeros, embeddings[:, 0, :], embeddings[:, 1, :]],
        dim=1,
    )
