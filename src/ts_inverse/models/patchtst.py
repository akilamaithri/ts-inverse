"""PatchTST victim, ported from the authors' released code.

Source: https://github.com/yuqinie98/PatchTST (Apache-2.0),
        PatchTST_supervised/layers/PatchTST_backbone.py and PatchTST_layers.py,
paper:  Nie et al., "A Time Series is Worth 64 Words", ICLR 2023, arXiv:2211.14730.

PORTED RATHER THAN IMPORTED because the attack differentiates *through* the
gradient: attack_dlg_invg_dia_worker.py and attack_ts_inverse_worker.py both call
torch.autograd.grad(..., create_graph=True) and then backprop through the result.
HuggingFace's PatchTSTModel routes attention through F.scaled_dot_product_attention,
whose fused kernels do not support double backward on every path. The upstream
_ScaledDotProductAttention below is explicit torch.matmul + F.softmax, which does.
Compute nodes also have no network, so a new dependency would have to be resolved
into venv-cu without disturbing the pinned torch==2.10.0+cu128.

THREE DEVIATIONS FROM UPSTREAM, all deliberate (see experiments/table2.py PROVENANCE):

  revin=False        Upstream defaults to True. RevIN subtracts the instance
                     mean/std *inside* the model, so the private window would be
                     recoverable only up to an affine map -- reintroducing exactly
                     the scale degeneracy already diagnosed for InvG/DIA, and
                     confounding "does patching resist" with "does normalisation
                     resist". The harness already min-max normalises the dataset.

  norm='LayerNorm'   Upstream defaults to 'BatchNorm'. The attack runs at batch
                     size 1, where BatchNorm1d is degenerate and its gradient
                     carries batch statistics that leak on a different channel.
                     Upstream supports LayerNorm as an option.

  dropout=0.0        Matches how FCN and CNN are configured in this harness
                     (BASE["dropout"] = 0) and keeps the victim deterministic.

Everything else -- patch_len/stride, learnable positional encoding, three encoder
layers, d_ff = 2*d_model, residual attention, GELU, flatten head -- is upstream's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaledDotProductAttention(nn.Module):
    """Upstream's explicit attention. Kept explicit for double-backward safety."""

    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attention=True):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        # A PLAIN FLOAT, not upstream's nn.Parameter(..., requires_grad=lsa). A
        # requires_grad=False Parameter still appears in model.parameters(), and
        # the attack calls torch.autograd.grad(y, model.parameters()), which raises
        # "One of the differentiated Tensors does not require grad" on it. Same
        # arithmetic, and it keeps the gradient list aligned with the trainable set.
        self.scale = (d_model // n_heads) ** -0.5

    def forward(self, q, k, v, prev=None):
        # q: [bs x n_heads x q_len x d_k], k: [bs x n_heads x d_k x q_len]
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, proj_dropout=0.0,
                 qkv_bias=True, res_attention=True):
        super().__init__()
        d_k = d_v = d_model // n_heads
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.res_attention = res_attention

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout, res_attention)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, x, prev=None):
        bs = x.size(0)
        q = self.W_Q(x).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(x).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q, k, v, prev=prev)
        else:
            output, attn_weights = self.sdp_attn(q, k, v)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, activation="gelu",
                 res_attention=True, pre_norm=False):
        super().__init__()
        self.res_attention = res_attention
        self.pre_norm = pre_norm

        self.self_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=0.0,
                                             proj_dropout=dropout, res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)  # deviation: upstream default is BatchNorm

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True), act, nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None):
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, _, scores = self.self_attn(src, prev=prev)
        else:
            src2, _ = self.self_attn(src)
            scores = None
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src = src + self.dropout_ffn(self.ff(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return (src, scores) if self.res_attention else src


class _TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation, n_layers, res_attention=True):
        super().__init__()
        self.res_attention = res_attention
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(d_model, n_heads, d_ff, dropout, activation, res_attention)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        output, scores = src, None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
        else:
            for mod in self.layers:
                output = mod(output)
        return output


class PatchTST_Predictor(nn.Module):
    """Channel-independent PatchTST with the harness's victim contract.

    Constructor keys are supplied by attack_dlg_invg_dia_worker._init_attack_worker_process
    from {**MODELS[name], **fam}. Two constraints on the names:

      - NOTHING may start with "input_" or "output_": that worker multiplies any
        such key by freq_in_day (=96). Hence patch_len / stride / d_ff.
      - `hidden_size` is the harness's width knob for every other victim, so it IS
        d_model here rather than a fourth spelling of the same idea.

    `self.name` is set per-instance and carries the patch config, because
    attack_learning_to_invert_worker keys its multi-GB gradient cache on model.name
    alone -- two patch lengths sharing a name would share one cache file. It also
    must not contain "TCN", which three workers use to gate the dropout-mask branch.
    """

    name = "PatchTST_Predictor"  # class-level, for models.model_classes

    def __init__(self, features=[0], hidden_size=64, input_size=24 * 4, output_size=24 * 4,
                 patch_len=16, stride=8, e_layers=3, n_heads=8, d_ff=None,
                 dropout=0.0, activation="gelu", pad_end=True):
        super().__init__()
        self.features = features
        d_model = hidden_size
        d_ff = d_ff if d_ff is not None else 2 * d_model
        if d_model % n_heads:
            raise ValueError(f"hidden_size {d_model} must be divisible by n_heads {n_heads}")
        if patch_len > input_size:
            raise ValueError(f"patch_len {patch_len} exceeds the observation window {input_size}")

        self.patch_len, self.stride, self.pad_end = patch_len, stride, pad_end
        patch_num = (input_size - patch_len) // stride + 1
        if pad_end:
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num

        self.W_P = nn.Linear(patch_len, d_model)
        # upstream positional_encoding(pe='zeros', learn_pe=True, ...)
        W_pos = torch.empty(patch_num, d_model)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        self.encoder = _TSTEncoder(d_model, n_heads, d_ff, dropout, activation, e_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(d_model * patch_num, output_size)
        self.head_dropout = nn.Dropout(0.0)

        self.name = f"PatchTST-P{patch_len}S{stride}-d{d_model}_Predictor"
        self.extra_info = {
            "patch_len": patch_len,
            "patch_stride": stride,
            "patch_num": patch_num,
            "patch_pad_end": pad_end,
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": n_heads,
            "e_layers": e_layers,
            "revin": False,
            "encoder_norm": "LayerNorm",
            "n_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(self, x):
        """(batch, seq_len, n_features) -> (batch, output_size).

        Channel-independent: every variable is patched and encoded with the same
        weights. The harness's target is univariate (batch_targets[:, :, 0]), so the
        first variable's head output is returned, matching FCN/CNN's (B, output_size).
        """
        z = x.transpose(1, 2)                                   # [bs x nvars x seq_len]
        n_vars = z.shape[1]
        if self.pad_end:
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [bs x nvars x patch_num x patch_len]

        u = self.W_P(z)                                         # [bs x nvars x patch_num x d_model]
        u = u.reshape(u.shape[0] * u.shape[1], u.shape[2], u.shape[3])
        u = self.dropout(u + self.W_pos)                        # [(bs*nvars) x patch_num x d_model]

        z = self.encoder(u)
        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])      # [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)                                # [bs x nvars x d_model x patch_num]

        z = self.head_dropout(self.head(self.flatten(z)))        # [bs x nvars x output_size]
        return z[:, 0, :]
