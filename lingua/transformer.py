# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)
from xformers.ops import AttentionBias, fmha

flex_attention_comp = torch.compile(flex_attention)


# =============================================================================
# ACTIVATION REGISTRY
# =============================================================================

ACTIVATION_REGISTRY: Dict[str, Callable] = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


def register_activation(name: str, fn: Callable):
    """Register a new activation function."""
    ACTIVATION_REGISTRY[name] = fn


# =============================================================================
# NORMALIZATION REGISTRY
# =============================================================================

class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


NORM_REGISTRY: Dict[str, Type[nn.Module]] = {
    "rmsnorm": RMSNorm,
    "layernorm": nn.LayerNorm,
}


def register_norm(name: str, cls: Type[nn.Module]):
    """Register a new normalization layer class."""
    NORM_REGISTRY[name] = cls


# =============================================================================
# POSITIONAL EMBEDDING REGISTRY
# =============================================================================

def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    """Implementation due to gpt-fast team:
    https://github.com/pytorch-labs/gpt-fast
    """
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_scaling: Optional[dict] = None,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


class RotaryEmbedding(torch.nn.Module):
    """RotaryEmbedding Module"""

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_scaling = rope_scaling

        self.register_buffer(
            "freqs_cis",
            torch.empty((self.max_seqlen, self.head_dim // 2, 2, 2)),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            rope_scaling=self.rope_scaling
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class NoPosEmbed(nn.Module):
    """No positional embedding (for ablations)."""
    embed_type = "none"  # Indicates this returns None

    def __init__(self, **kwargs):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None):
        return None


class ALiBiEmbedding(nn.Module):
    """
    ALiBi: Attention with Linear Biases (Press et al., 2021)
    https://arxiv.org/abs/2108.12409

    Instead of adding positional embeddings to tokens, ALiBi adds a static,
    non-learned bias to attention scores: bias[i,j] = -m * |i - j|
    where m is a head-specific slope.

    Slopes are computed as a geometric sequence:
    - For n_heads being power of 2: m_i = 2^(-8/n * i) for i in [1, n]
    - For non-power of 2: combine slopes from closest power of 2 and half of that

    Reference: https://github.com/ofirpress/attention_with_linear_biases
    """
    embed_type = "alibi"  # Indicates this returns attention bias

    def __init__(
        self,
        n_heads: int,
        max_seqlen: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.max_seqlen = max_seqlen

        # Compute slopes following the paper's geometric sequence
        slopes = self._get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)

        # Precompute bias matrix for max_seqlen
        # Will be computed in reset_parameters
        self.register_buffer(
            "bias",
            torch.zeros((max_seqlen, max_seqlen, n_heads)),
            persistent=False,
        )

    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute head-specific slopes for ALiBi.

        For n_heads = 8: slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
                                = 2^(-1), 2^(-2), ..., 2^(-8)

        The ratio is 2^(-8/n_heads) raised to powers 1, 2, ..., n_heads.
        """
        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])

        # Check if n_heads is power of 2
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            # For non-power of 2, use closest power of 2 and half of it
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_p2 = get_slopes_power_of_2(closest_power_of_2)

            # Get additional slopes at half the base (skip every other)
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            extra_slopes = torch.tensor([
                extra_base * (extra_base ** (2 * i))
                for i in range(n_heads - closest_power_of_2)
            ])

            return torch.cat([slopes_p2, extra_slopes])

    def reset_parameters(self):
        """Precompute the ALiBi bias matrix."""
        # Create distance matrix: distance[i, j] = |i - j|
        positions = torch.arange(self.max_seqlen)
        # For causal attention, we use: bias[i, j] = -m * (i - j) for j <= i
        # This penalizes attending to tokens further in the past
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [max_seqlen, max_seqlen]

        # bias[i, j, h] = -slopes[h] * distance[i, j]
        # Shape: [max_seqlen, max_seqlen, n_heads]
        self.bias[...] = -distance.unsqueeze(-1) * self.slopes.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        seqlen: Optional[int] = None,
        tok_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns ALiBi bias to be added to attention scores.

        Returns:
            Tensor of shape [seqlen, seqlen, n_heads] to be added to attention logits
            before softmax. For SDPA, will need to be reshaped to [1, n_heads, seqlen, seqlen].
        """
        if seqlen is not None:
            return self.bias[:seqlen, :seqlen, :]
        elif tok_idx is not None:
            # Handle token indices for incremental decoding
            return self.bias[tok_idx, :tok_idx.max()+1, :]
        else:
            return self.bias


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embeddings from "Attention is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    These are added to token embeddings before the transformer layers.
    """
    embed_type = "additive"  # Indicates this returns additive embedding

    def __init__(
        self,
        dim: int,
        max_seqlen: int = 1024,
        base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.base = base

        # Precompute sinusoidal embeddings
        self.register_buffer(
            "embeddings",
            torch.zeros((max_seqlen, dim)),
            persistent=False,
        )

    def reset_parameters(self):
        """Precompute sinusoidal positional embeddings."""
        position = torch.arange(self.max_seqlen).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.dim, 2).float() * -(math.log(self.base) / self.dim)
        )

        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        self.embeddings[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        if self.dim % 2 == 0:
            self.embeddings[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle odd dimensions
            self.embeddings[:, 1::2] = torch.cos(position * div_term[:-1])

    def forward(
        self,
        seqlen: Optional[int] = None,
        tok_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns sinusoidal embeddings to be added to token embeddings.

        Returns:
            Tensor of shape [seqlen, dim] to be added to token embeddings.
        """
        if seqlen is not None:
            return self.embeddings[:seqlen, :]
        elif tok_idx is not None:
            return self.embeddings[tok_idx, :]
        else:
            return self.embeddings


class LearnedEmbedding(nn.Module):
    """
    Learned Absolute Positional Embeddings (used in BERT, GPT-2, etc.)

    Each position has a learnable embedding vector that is added to token embeddings.
    This is essentially a lookup table indexed by position.

    Note: "one-hot" refers to the fact that positions are one-hot encoded indices
    into a learnable embedding matrix.
    """
    embed_type = "additive"  # Indicates this returns additive embedding

    def __init__(
        self,
        dim: int,
        max_seqlen: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen

        # Learnable position embeddings
        self.embeddings = nn.Parameter(torch.zeros(max_seqlen, dim))

    def reset_parameters(self):
        """Initialize position embeddings with truncated normal."""
        init_std = self.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.embeddings,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def forward(
        self,
        seqlen: Optional[int] = None,
        tok_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns learned embeddings to be added to token embeddings.

        Returns:
            Tensor of shape [seqlen, dim] to be added to token embeddings.
        """
        if seqlen is not None:
            return self.embeddings[:seqlen, :]
        elif tok_idx is not None:
            return self.embeddings[tok_idx, :]
        else:
            return self.embeddings


# Add embed_type markers to existing classes
RotaryEmbedding.embed_type = "rotary"
NoPosEmbed.embed_type = "none"


POSEMBED_REGISTRY: Dict[str, Type[nn.Module]] = {
    "rope": RotaryEmbedding,
    "none": NoPosEmbed,
    "nope": NoPosEmbed,  # Alias for "no positional embedding"
    "alibi": ALiBiEmbedding,
    "sinusoidal": SinusoidalEmbedding,
    "learned": LearnedEmbedding,
    "onehot": LearnedEmbedding,  # Alias for learned embeddings
}


def register_posembed(name: str, cls: Type[nn.Module]):
    """Register a new positional embedding class."""
    POSEMBED_REGISTRY[name] = cls


# =============================================================================
# INIT STD FACTOR
# =============================================================================

class InitStdFactor(Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


# =============================================================================
# BASE TRANSFORMER ARGS
# =============================================================================

@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    qk_norm: bool = False

    hidden_dim: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256

    norm_eps: float = 1e-5

    # Registry-based selections
    norm_type: str = "rmsnorm"
    activation: str = "silu"
    pos_embed_type: str = "rope"

    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    doc_id = torch.repeat_interleave(lengths)
    doc_start = lengths_to_start_ids(lengths)
    doc_start = doc_start[doc_id]
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start
    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# =============================================================================
# ATTENTION
# =============================================================================

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            norm_cls = NORM_REGISTRY.get(norm_type, RMSNorm)
            self.q_norm = norm_cls(head_dim, eps=qk_norm_eps)
            self.k_norm = norm_cls(head_dim, eps=qk_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings if available
        if freq_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq, xk, xv, is_causal=is_causal, attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

        output = self.wo(output.reshape(output_shape))
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)

        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std / factor, a=-3 * init_std, b=3 * init_std)

        if self.qk_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()


# =============================================================================
# FEEDFORWARD
# =============================================================================

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        force_hidden_dim_value: bool = False,
        mp_size: int = 1,
        activation: str = "silu",
    ):
        super().__init__()

        if not force_hidden_dim_value:
            hidden_dim = int(2 * hidden_dim / 3)
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activation_fn = ACTIVATION_REGISTRY.get(activation, F.silu)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(self.activation_fn(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(w.weight, mean=0.0, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=out_init_std, a=-3 * out_init_std, b=3 * out_init_std)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (args.n_heads is not None), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        norm_cls = NORM_REGISTRY.get(args.norm_type, RMSNorm)

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            qk_norm=args.qk_norm,
            qk_norm_eps=args.norm_eps,
            norm_type=args.norm_type,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else 4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            force_hidden_dim_value=args.hidden_dim is not None,
            activation=args.activation,
        )
        self.attention_norm = norm_cls(args.dim, eps=args.norm_eps)
        self.ffn_norm = norm_cls(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x), freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


# =============================================================================
# BASE TRANSFORMER
# =============================================================================

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.pos_embed_type = args.pos_embed_type
        self.n_heads = args.n_heads or args.dim // args.head_dim

        # Build positional embeddings from registry
        posembed_cls = POSEMBED_REGISTRY.get(args.pos_embed_type, RotaryEmbedding)

        # Initialize based on positional embedding type
        if args.pos_embed_type == "rope":
            self.pos_embeddings = posembed_cls(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=args.max_seqlen,
                rope_scaling=args.rope_scaling,
            )
        elif args.pos_embed_type == "alibi":
            self.pos_embeddings = posembed_cls(
                n_heads=self.n_heads,
                max_seqlen=args.max_seqlen,
            )
        elif args.pos_embed_type in ("sinusoidal", "learned", "onehot"):
            self.pos_embeddings = posembed_cls(
                dim=args.dim,
                max_seqlen=args.max_seqlen,
            )
        else:
            # none/nope
            self.pos_embeddings = posembed_cls()

        # Store the embed type for fast dispatch
        self._posembed_type = getattr(self.pos_embeddings, 'embed_type', 'none')

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen, dim = h.shape
        freq_cis = None

        # Handle different positional embedding types
        if self._posembed_type == "rotary":
            # RoPE: get frequency tensor for rotary embeddings
            freq_cis = self.pos_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        elif self._posembed_type == "alibi":
            # ALiBi: get attention bias and combine with mask
            alibi_bias = self.pos_embeddings(seqlen=seqlen, tok_idx=tok_idx)
            # alibi_bias shape: [seqlen, seqlen, n_heads]
            # For SDPA, we need [1, n_heads, seqlen, seqlen]
            # Move to same device as h and convert dtype
            alibi_mask = alibi_bias.to(device=h.device, dtype=h.dtype).permute(2, 0, 1).unsqueeze(0)

            # Combine with causal mask
            if attn_impl == "sdpa":
                # Create causal mask (upper triangular = -inf)
                causal = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), device=h.device, dtype=h.dtype),
                    diagonal=1
                )
                # Add ALiBi bias to causal mask
                # ALiBi bias is 0 for same position, negative for attending to past
                mask = causal.unsqueeze(0).unsqueeze(0) + alibi_mask
            elif attn_impl == "fmha":
                # For xformers fmha, create alibi as a tensor bias
                # xformers expects [B, H, Q, K] or broadcastable
                causal = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), device=h.device, dtype=h.dtype),
                    diagonal=1
                )
                alibi_with_causal = causal.unsqueeze(0).unsqueeze(0) + alibi_mask
                # Expand for batch size
                mask = alibi_with_causal.expand(bsz, -1, -1, -1)

        elif self._posembed_type == "additive":
            # Sinusoidal or Learned: add to hidden states
            pos_emb = self.pos_embeddings(seqlen=seqlen, tok_idx=tok_idx)
            # pos_emb shape: [seqlen, dim]
            # Move to same device as h and convert dtype
            h = h + pos_emb.unsqueeze(0).to(device=h.device, dtype=h.dtype)

        # elif self._posembed_type == "none": pass (no positional encoding)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        self.pos_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
