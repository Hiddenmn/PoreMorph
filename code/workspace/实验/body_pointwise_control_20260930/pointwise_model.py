"""Same-descriptor residual point MLP: no token interaction before pooling."""
from pathlib import Path
import sys

import torch
from torch import nn

PARENT = Path(__file__).resolve().parent.parent / "body_interface_decoder_20260930"
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from model import BodyInterfaceDecoder, POINTS, TYPES, FEATURES, SUMMARY


class PointwiseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.pointwise = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.norm2 = nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, x, valid):
        x = x + self.pointwise(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x * valid[..., None]


class PointwiseBodyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(FEATURES, 32)
        self.type_embedding = nn.Parameter(torch.randn(TYPES, 32) * .02)
        self.blocks = nn.ModuleList([PointwiseBlock() for _ in range(2)])
        self.classifier = nn.Sequential(nn.Linear(32 * TYPES + SUMMARY, 64), nn.GELU(), nn.Linear(64, 5))
        self.register_buffer("type_ids", torch.arange(TYPES).repeat_interleave(POINTS), persistent=False)

    def forward(self, points, valid, summary):
        b, n, c = points.shape
        if (n, c) != (TYPES * POINTS, FEATURES) or summary.shape != (b, SUMMARY):
            raise ValueError("Unexpected body descriptor shape")
        valid = valid.bool()
        x = (self.input(points) + self.type_embedding[self.type_ids][None]) * valid[..., None]
        for block in self.blocks:
            x = block(x, valid)
        mask = valid.reshape(b, TYPES, POINTS, 1)
        pooled = (x.reshape(b, TYPES, POINTS, 32) * mask).sum(2) / mask.sum(2).clamp_min(1)
        base = summary[:, 3:8].clamp_min(1e-8)
        base = (base / base.sum(-1, keepdim=True)).log()
        return base + self.classifier(torch.cat((pooled.flatten(1), summary), 1))


def initialise():
    """Original untrained shared state; deterministic new replacement weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original = BodyInterfaceDecoder("separate")
        control = PointwiseBodyDecoder()
        old, current = original.state_dict(), control.state_dict()
        shared = {k: v for k, v in old.items() if k in current}
        for key, value in shared.items():
            assert value.shape == current[key].shape
            current[key].copy_(value)
        control.load_state_dict(current, strict=True)
    assert sum(p.numel() for p in control.parameters()) == 25509
    assert sum(v.numel() for v in shared.values()) == 17125
    return control, shared
