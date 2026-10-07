"""Coordinate-aligned native-image decoding, without label-derived inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class PixelNorm(nn.Module):
    """Normalize channels per pixel: no crop-size-dependent spatial statistics."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class Block(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(incoming, width, 3, padding=1), PixelNorm(width), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1), PixelNorm(width), nn.GELU())

    def forward(self, x):
        return self.net(x)


def sample_native_grid(source, origins, side, stride=1, native_side=1024):
    """Pixel-center coordinates; equals align_corners=False resize then crop."""
    count = side // stride
    offsets = (torch.arange(count, device=source.device, dtype=torch.float32) + .5) * stride
    yy = (origins[:, 0, None] + offsets[None]) * (2. / native_side) - 1
    xx = (origins[:, 1, None] + offsets[None]) * (2. / native_side) - 1
    grid = torch.stack((xx[:, None, :].expand(-1, count, -1), yy[:, :, None].expand(-1, -1, count)), -1)
    return F.grid_sample(source.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


class NativeDetailDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = Block(1, 16)
        self.middle = Block(16, 32)
        self.deep = Block(32, 64)
        self.context_projection = nn.Conv2d(32, 64, 1)
        self.context_gate = nn.Conv2d(64 + 32, 64, 1)
        self.up_middle = Block(64 + 32, 32)
        self.up_fine = Block(32 + 16, 16)
        self.output = nn.Conv2d(16, 8, 1)
        # A small NONZERO output propagates gradients to every branch initially.
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, coarse_logits, context):
        fine = self.fine(image)
        middle = self.middle(F.avg_pool2d(fine, 2))
        deep = self.deep(F.avg_pool2d(middle, 2))
        if context.shape[-2:] != deep.shape[-2:]:
            raise ValueError("Global context must be sampled onto the local quarter grid")
        gate = self.context_gate(torch.cat((deep, context), 1)).sigmoid()
        deep = deep + gate * self.context_projection(context)
        middle = self.up_middle(torch.cat((F.interpolate(deep, size=middle.shape[-2:],
                                                        mode="bilinear", align_corners=False), middle), 1))
        fine = self.up_fine(torch.cat((F.interpolate(middle, size=fine.shape[-2:],
                                                    mode="bilinear", align_corners=False), fine), 1))
        return coarse_logits + self.output(fine)
