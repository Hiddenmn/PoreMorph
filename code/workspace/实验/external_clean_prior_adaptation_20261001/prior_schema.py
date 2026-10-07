"""Fixed physical-condition priors and append-only character coverage.

Only source metadata assigned to the target TRAIN sets defines the new
characters. No image, label, validation score or test artifact is read here.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F

PROMPTS = {
    'bentheimer': ('岩心：Bentheimer砂岩；润湿性：强水湿；前序：排驱；'
                   '当前：稳态渗吸阶段，注入油相分流比为0；'
                   '水相：3.5 wt%碘化钾盐水；油相：癸烷。'),
    'wf1': ('岩心：Ketton石灰岩；润湿性：水湿；前序：盐水饱和后注油；'
            '当前：第一次水驱后；水相：碘化钾盐水；油相：碘代癸烷标记的癸烷。'),
}
SCHEMA_VERSION = 'external_physical_conditions_v1_20261001'
NEW_ROW_SEED = 4201001
NEW_ROW_NOISE_STD = 0.01
SOURCES = {
    'bentheimer': {
        'dataset_doi': '10.17612/KT0B-SZ28',
        'metadata': 'https://api.datacite.org/dois/10.17612/KT0B-SZ28',
        'paper': 'https://doi.org/10.1029/2019WR026396',
        'stage_basis': 'Archived training files Core_1_imbFo0_3_45_zoom; repository naming convention.',
        'derived_field': 'ImbFo0 means imbibition with oil injection fractional flow 0, NOT oil saturation 0.',
    },
    'wf1': {
        'dataset_doi': '10.17612/P7HT11',
        'metadata': 'https://api.datacite.org/dois/10.17612/P7HT11',
        'paper': 'https://doi.org/10.1016/j.advwatres.2018.09.010',
        'stage_basis': 'Archived training WF1 files; repository sequence OI -> WF1 -> GI -> WF2.',
        'excluded_later_stage': 'Nitrogen injection belongs to GI, after WF1.',
    },
}


def extend_tokenizer(original):
    """Keep every existing ID; append a sorted union of TRAIN-description characters."""
    tokenizer = copy.deepcopy(original)
    characters = set(''.join(PROMPTS.values()))
    appended = sorted(characters.difference(original.token2idx))
    tokenizer.idx2token = list(original.idx2token) + appended
    tokenizer.token2idx = {token: i for i, token in enumerate(tokenizer.idx2token)}
    assert tokenizer.idx2token[:original.vocab_size] == original.idx2token
    assert (tokenizer.pad_id, tokenizer.unk_id, tokenizer.cls_id) == (
        original.pad_id, original.unk_id, original.cls_id)
    for prompt in PROMPTS.values():
        ids, mask = tokenizer.encode(prompt)
        assert len(prompt) + 1 <= tokenizer.max_len
        assert int(mask.sum()) == len(prompt) + 1
        assert not bool((ids[mask.bool()] == tokenizer.unk_id).any())
    return tokenizer


def token_stats(tokenizer, text):
    ids, mask = tokenizer.encode(text)
    active = ids[mask.bool()]
    return {
        'characters': len(text),
        'active_tokens_including_cls': int(mask.sum()),
        'unknown_active_tokens': int((active == tokenizer.unk_id).sum()),
        'unknown_characters_full_text': sorted(set(text).difference(tokenizer.token2idx)),
        'truncated_characters': max(0, len(text) + 1 - tokenizer.max_len),
    }


class AppendOnlyEmbedding(nn.Module):
    """Freeze original vectors exactly; learn only newly added character rows.

    New vectors start near the old UNK vector using a local, deterministic RNG.
    They are not pretrained semantic embeddings. Existing rows never enter the
    optimizer, including weight decay.
    """
    def __init__(self, old_weight, new_count, unk_id=1, padding_idx=0):
        super().__init__()
        if new_count <= 0:
            raise ValueError('Expected positive append-only vocabulary extension')
        self.padding_idx = padding_idx
        self.old_weight = nn.Parameter(old_weight.detach().clone(), requires_grad=False)
        generator = torch.Generator(device='cpu').manual_seed(NEW_ROW_SEED)
        noise = torch.randn(new_count, old_weight.shape[1], generator=generator,
                            dtype=torch.float32) * NEW_ROW_NOISE_STD
        initial = old_weight[unk_id].detach().cpu().float()[None] + noise
        self.new_weight = nn.Parameter(initial.to(device=old_weight.device, dtype=old_weight.dtype))
        self.num_embeddings = old_weight.shape[0] + new_count
        self.embedding_dim = old_weight.shape[1]

    @property
    def weight(self):
        return torch.cat((self.old_weight, self.new_weight), dim=0)

    def forward(self, ids):
        return F.embedding(ids, self.weight, padding_idx=self.padding_idx)
