"""Shared coarse/fine supervision, independent of model output parameterization.

Fine labels: bg, rock, water, film, corner, droplet, columnar, clustered (0..7).
Coarse labels: bg, rock, water, oil (0..3). A batch's explicit granularity flag
determines semantics; coarse oil (3) must NEVER be treated as fine film (3).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RAW_TO_PHASE = {"wf1": (0, 3, 2, 1), "bentheimer": (0, 2, 3, 1)}


def remap_external_labels(raw, dataset):
    if dataset not in RAW_TO_PHASE:
        raise ValueError(f"Unknown label schema: {dataset}")
    if raw.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Expected integer source label IDs")
    if raw.numel() == 0 or raw.min().item() < 0 or raw.max().item() > 3:
        raise ValueError("Unrecognized source labels; refusing silent clipping")
    mapping = torch.tensor(RAW_TO_PHASE[dataset], dtype=torch.long, device=raw.device)
    return mapping[raw.long()]


def phase_marginal_logp(eight_logits):
    if eight_logits.ndim != 4 or eight_logits.shape[1] != 8:
        raise ValueError("Expected Bx8xHxW logits or normalized fine log probabilities")
    logp = F.log_softmax(eight_logits.float(), dim=1)
    return torch.cat([logp[:, :3], torch.logsumexp(logp[:, 3:], dim=1, keepdim=True)], dim=1)


def weighted_dice_from_logp(logp, target, weights):
    probs = logp.exp()
    truth = F.one_hot(target, num_classes=logp.shape[1]).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)
    intersection = (probs * truth).sum(dimensions)
    union = (probs + truth).sum(dimensions)
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return 1 - (dice * weights).sum() / (weights.sum() + 1e-8)


class MixedGranularityLoss(nn.Module):
    """Same marginal loss for a flat head and a soft hierarchical head.

    Per-granularity CE/Dice losses are mixed by sample count. Class weights must
    be estimated from the corresponding TRAINING annotations, not val/test.
    Dice remains a group-batch loss; this is not a per-pixel likelihood identity.
    """
    def __init__(self, fine_weights, phase_weights):
        super().__init__()
        fine_weights = torch.as_tensor(fine_weights, dtype=torch.float32)
        phase_weights = torch.as_tensor(phase_weights, dtype=torch.float32)
        if fine_weights.shape != (8,) or phase_weights.shape != (4,):
            raise ValueError("Expected 8 fine and 4 phase weights")
        if not (torch.isfinite(fine_weights).all() and torch.isfinite(phase_weights).all()
                and (fine_weights > 0).all() and (phase_weights > 0).all()):
            raise ValueError("Weights must be finite and strictly positive")
        self.register_buffer("fine_weights", fine_weights)
        self.register_buffer("phase_weights", phase_weights)

    def forward(self, eight_logits, target, is_coarse):
        if eight_logits.ndim != 4 or eight_logits.shape[1] != 8:
            raise ValueError("Expected eight-class outputs")
        if target.shape != (eight_logits.shape[0], *eight_logits.shape[2:]) or target.dtype != torch.long:
            raise ValueError("Target shape/dtype mismatch")
        if is_coarse.shape != (eight_logits.shape[0],) or is_coarse.dtype != torch.bool:
            raise ValueError("Explicit boolean per-image granularity flag required")
        if target.device != eight_logits.device or is_coarse.device != eight_logits.device:
            raise ValueError("All inputs must be on the same device")
        total = eight_logits.sum() * 0
        for coarse, weights in ((False, self.fine_weights), (True, self.phase_weights)):
            selected = is_coarse if coarse else ~is_coarse
            count = int(selected.sum())
            if not count:
                continue
            targets = target[selected]
            classes = 4 if coarse else 8
            if targets.min().item() < 0 or targets.max().item() >= classes:
                raise ValueError("Label IDs violate their explicit granularity")
            logits = eight_logits[selected]
            logp = phase_marginal_logp(logits) if coarse else F.log_softmax(logits.float(), dim=1)
            ce = F.nll_loss(logp, targets, weight=weights)
            dice = weighted_dice_from_logp(logp, targets, weights)
            total = total + (count / eight_logits.shape[0]) * (0.5 * ce + 0.5 * dice)
        return total
