"""Target adaptation copies; frozen source artifacts never receive writes."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

import common as c
from prior_schema import extend_tokenizer, AppendOnlyEmbedding


class AdaptationModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in c.ARMS:
            raise ValueError(arm)
        self.arm = arm
        files = c.locked.paths(c.SEED)
        c.order.base.set_seed(c.SEED)
        self.tokenizer = c.order.base.CharTokenizer.load(files['tokenizer'])
        self.global_model = c.order.build_model('fusion_before_ppm', self.tokenizer.vocab_size)
        source = torch.load(files['global_checkpoint'], map_location='cpu', weights_only=False)
        self.global_model.load_state_dict(source['model_state_dict'], strict=True)
        self.global_model.text_encoder.requires_grad_(False)
        legacy_vocab_size = self.tokenizer.vocab_size
        self.tokenizer = extend_tokenizer(self.tokenizer)
        self.global_model.text_encoder.token_embed = AppendOnlyEmbedding(
            self.global_model.text_encoder.token_embed.weight,
            self.tokenizer.vocab_size - legacy_vocab_size,
            unk_id=self.tokenizer.unk_id, padding_idx=self.tokenizer.pad_id)
        if arm == 'native_detail':
            self.native_model = c.NativeNoContext()
            source = torch.load(files['native_checkpoint'], map_location='cpu', weights_only=False)
            self.native_model.load_state_dict(source['model'], strict=True)
        else:
            self.native_model = None

    def train(self, mode=True):
        super().train(mode)
        self.global_model.text_encoder.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, native_image, global_image, ids, attention):
        assert native_image.ndim == global_image.ndim == 4 and native_image.shape[:2] == (1, 1)
        coarse = self.global_model(global_image, ids, attention)
        # Cast preserves the existing BF16 quantization while retaining gradients.
        up = c.project_native(coarse, *native_image.shape[-2:])
        logits = up if self.native_model is None else self.native_model(native_image, up, None)
        return coarse, logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_state(self):
        return frozen_from_state(self.state_dict())


def frozen_from_state(state):
    prefix = 'global_model.text_encoder.'
    result = {'text.' + key.removeprefix(prefix): value for key, value in state.items()
              if key.startswith(prefix) and key != prefix + 'token_embed.new_weight'}
    result.update({key: value for key, value in state.items()
                   if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))})
    return result


def phase_loss(logits, target, weights):
    logp = c.phase_marginal_logp(logits)
    return .5*F.nll_loss(logp, target, weight=weights) + .5*c.weighted_dice_from_logp(logp, target, weights)


def to_device(sample):
    image, low, target = sample
    return (torch.from_numpy(image)[None, None].cuda(),
            torch.from_numpy(low)[None, None].repeat(1, 3, 1, 1).cuda().to(memory_format=torch.channels_last),
            torch.from_numpy(target.astype('int64'))[None].cuda())
