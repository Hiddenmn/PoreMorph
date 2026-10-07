"""Structural feature-path removal; the multimodal coarse logits are retained."""
from pathlib import Path
import sys
import hashlib
import torch
from torch.nn import functional as F

ORIGINAL = Path(__file__).resolve().parent.parent / "native_detail_decoder_20260930"
sys.path.insert(0, str(ORIGINAL))
from architecture import NativeDetailDecoder


def tensor_digest(state):
    return hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                                   for t in state.values())).hexdigest()


class NativeNoContext(NativeDetailDecoder):
    def __init__(self):
        # Consume the identical RNG sequence before removal, including the small
        # output initialization, so all shared layers exactly match the full arm.
        super().__init__()
        self.full_initial_sha256 = tensor_digest(self.state_dict())
        del self.context_projection
        del self.context_gate

    def forward(self, image, coarse_logits, context):
        # `context` is intentionally not read. Keeping the call signature lets
        # data, geometry, augmentations, loss and evaluation remain identical.
        fine = self.fine(image)
        middle = self.middle(F.avg_pool2d(fine, 2))
        deep = self.deep(F.avg_pool2d(middle, 2))
        middle = self.up_middle(torch.cat((F.interpolate(deep, size=middle.shape[-2:],
                                                        mode="bilinear", align_corners=False), middle), 1))
        fine = self.up_fine(torch.cat((F.interpolate(middle, size=fine.shape[-2:],
                                                    mode="bilinear", align_corners=False), fine), 1))
        return coarse_logits + self.output(fine)
