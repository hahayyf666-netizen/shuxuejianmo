"""Small three branch, two head Q3 prediction models.

The forward signature deliberately excludes all interpretation metadata.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Q3Model(nn.Module):
    def __init__(self, variant: str = "B0", dropout: float = 0.2):
        super().__init__()
        if variant not in {"B0", "B1"}:
            raise ValueError("variant must be B0 or B1")
        self.variant = variant
        self.projections = nn.ModuleList([nn.Linear(dim, 64) for dim in (768, 74, 35)])
        self.activation = nn.GELU()
        self.encoders = nn.ModuleList(
            [nn.GRU(64, 32, batch_first=True, bidirectional=True, num_layers=1, dropout=0.0) for _ in range(3)]
        ) if variant == "B1" else None
        self.fusion = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(64, 3)
        self.regressor = nn.Linear(64, 1)

    @staticmethod
    def _check_inputs(xs: tuple[torch.Tensor, ...], validity_mask: torch.Tensor) -> None:
        if len(xs) != 3 or any(x.ndim != 3 for x in xs):
            raise ValueError("three [batch,50,dim] tensors required")
        batch = xs[0].shape[0]
        if any(x.shape != (batch, 50, dim) for x, dim in zip(xs, (768, 74, 35))):
            raise ValueError("feature shape mismatch")
        if any(x.dtype != torch.float32 for x in xs):
            raise TypeError("model features must be float32")
        if any(not bool(torch.isfinite(x).all()) for x in xs):
            raise ValueError("nonfinite feature")
        if validity_mask.shape != (batch, 50) or validity_mask.dtype != torch.bool:
            raise ValueError("validity_mask must be bool [batch,50]")
        if bool((validity_mask.sum(dim=1) == 0).any()):
            raise ValueError("zero content length")

    @staticmethod
    def _compact(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item())
        compact = x.new_zeros((x.shape[0], max_len, x.shape[2]))
        for row in range(x.shape[0]):
            compact[row, : int(lengths[row])] = x[row, mask[row]]
        return compact, lengths

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs = (text_feat, audio_feat, vision_feat)
        self._check_inputs(xs, validity_mask)
        summaries = []
        for i, (x, projection) in enumerate(zip(xs, self.projections)):
            projected = self.activation(projection(x))
            if self.variant == "B0":
                mask = validity_mask.unsqueeze(-1)
                pooled = (projected * mask).sum(dim=1) / validity_mask.sum(dim=1, keepdim=True)
            else:
                compact, lengths = self._compact(projected, validity_mask)
                packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
                encoded, _ = self.encoders[i](packed)
                unpacked, _ = pad_packed_sequence(encoded, batch_first=True, total_length=compact.shape[1])
                arange = torch.arange(compact.shape[1], device=x.device)[None, :]
                pooled = (unpacked * (arange < lengths[:, None]).unsqueeze(-1)).sum(dim=1) / lengths[:, None]
            summaries.append(pooled)
        fused = self.fusion(torch.cat(summaries, dim=1))
        return self.classifier(fused), 3.0 * torch.tanh(self.regressor(fused).squeeze(-1))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
