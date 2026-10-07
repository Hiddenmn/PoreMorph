"""Body morphology decisions with paired separate/coupled contact encoding."""
import torch
from torch import nn
from torch.nn import functional as F

POINTS = 16
TYPES = 3
FEATURES = 31
SUMMARY = 13


class ContactBlock(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width*3)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))

    def forward(self, x, allowed, valid):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c//self.heads).permute(2,0,3,1,4)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:,None], dropout_p=0.)
        x = x + self.projection(mixed.transpose(1,2).reshape(b,n,c))
        x = x + self.ffn(self.norm2(x))
        return x*valid[...,None]


class BodyInterfaceDecoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("separate", "coupled"):
            raise ValueError(mode)
        self.mode = mode
        self.input = nn.Linear(FEATURES,32)
        self.type_embedding = nn.Parameter(torch.randn(TYPES,32)*.02)
        self.blocks = nn.ModuleList([ContactBlock() for _ in range(2)])
        self.classifier = nn.Sequential(nn.Linear(32*TYPES+SUMMARY,64),nn.GELU(),nn.Linear(64,5))
        nn.init.normal_(self.classifier[-1].weight,std=.001)
        nn.init.zeros_(self.classifier[-1].bias)
        self.register_buffer("type_ids",torch.arange(TYPES).repeat_interleave(POINTS),persistent=False)

    def forward(self, points, valid, summary):
        # No targets, class counts, sample/stage IDs or absolute position here.
        b,n,c = points.shape
        if (n,c) != (TYPES*POINTS,FEATURES) or summary.shape != (b,SUMMARY):
            raise ValueError("Unexpected body descriptor shape")
        valid = valid.bool()
        same = self.type_ids[:,None] == self.type_ids[None,:]
        allowed = valid[:,None,:].expand(-1,n,-1).clone()
        if self.mode == "separate":
            allowed &= same[None]
        # Padded queries in empty sets may have no valid key. A private self
        # edge prevents all-masked softmax; padded outputs are then discarded.
        empty = ~allowed.any(-1)
        allowed |= empty[:,:,None] & torch.eye(n,device=points.device,dtype=torch.bool)[None]
        x = (self.input(points) + self.type_embedding[self.type_ids][None])*valid[...,None]
        for block in self.blocks:
            x = block(x,allowed,valid)
        x = x.reshape(b,TYPES,POINTS,32)
        mask = valid.reshape(b,TYPES,POINTS,1)
        pooled = (x*mask).sum(2)/mask.sum(2).clamp_min(1)
        base = summary[:,3:8].clamp_min(1e-8)
        base = (base/base.sum(-1,keepdim=True)).log()
        return base + self.classifier(torch.cat((pooled.flatten(1),summary),1))


def augment(points, generator):
    """D4 on body-relative positions and oil-mask gradients, not descriptors."""
    result = points.clone()
    b = len(result)
    flip = torch.randint(0,2,(b,1),generator=generator,device=points.device)*2-1
    rotation = torch.randint(0,4,(b,1),generator=generator,device=points.device)
    for first,second in ((0,1),(2,3)):
        x,y = result[:,:,first].clone()*flip,result[:,:,second].clone()
        xx = torch.where(rotation==0,x,torch.where(rotation==1,-y,torch.where(rotation==2,-x,y)))
        yy = torch.where(rotation==0,y,torch.where(rotation==1,x,torch.where(rotation==2,-y,-x)))
        result[:,:,first],result[:,:,second] = xx,yy
    return result


def body_loss(logits, counts, class_weights):
    counts = counts.float()
    target = counts[:,3:]
    weighted = target*class_weights
    ce = -(weighted*logits.float().log_softmax(-1)).sum()/weighted.sum().clamp_min(1)
    prob = logits.float().softmax(-1)
    intersection = (prob*target).sum(0)
    denominator = (prob*counts.sum(1,keepdim=True)).sum(0)+target.sum(0)
    dice = 1-(2*intersection+1e-6)/(denominator+1e-6)
    return .5*ce + .5*(dice*class_weights).sum()/class_weights.sum()
