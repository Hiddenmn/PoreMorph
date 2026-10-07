"""Frozen external Native inference -> shared body descriptors, TRAIN/VAL only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import body_common as c
import spatial as s


def digest_identity(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def cache_folder(dataset, smoke=False):
    if dataset not in c.DATASETS:
        raise ValueError(dataset)
    return c.HERE / ('smoke_cache' if smoke else 'cache') / dataset


def validate_shard(z, row, data, identity_digest):
    assert str(z['identity_sha256']) == identity_digest
    assert int(z['row']) == row and str(z['split']) == data.split
    assert str(z['label_id']) == data.entry['records'][row]['label_id']
    n = len(z['points'])
    for key, (dtype, shape) in c.LAYOUT.items():
        assert z[key].dtype == dtype and z[key].shape == (n, *shape), key
        assert np.isfinite(z[key]).all(), key
    assert z['components'].dtype == np.int32 and z['prediction'].dtype == np.uint8
    assert int(z['components'].max()) == n
    truth = np.asarray(data.masks[row])
    counts, outside, frozen = s.targets(z['components'], z['prediction'], truth)
    for grid in c.GRIDS:
        for suffix, values in (('counts', counts), ('outside', outside), ('frozen', frozen)):
            np.testing.assert_array_equal(z[grid + '_' + suffix], values[grid])
    choices = z['summary'][:, 3:8].argmax(1)
    s.verify_dense(z['components'], z['prediction'], truth, choices, counts, outside)
    return n


@torch.inference_mode()
def prepare(dataset, identity, smoke=False):
    c.stop_check()
    folder = cache_folder(dataset, smoke)
    ident_hash = digest_identity(identity)
    if (folder / 'manifest.json').exists():
        meta = c.read(folder / 'manifest.json')
        assert meta['identity'] == identity and meta['smoke'] == smoke
        if meta['status'] == 'completed':
            return verify_cache(folder, identity)
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'shards').mkdir()
        meta = dict(status='preparing', identity=identity, dataset=dataset,
                    smoke=smoke, test_pixels_read=False, created_utc=c.utc())
        c.write(folder / 'manifest.json', meta)
    upstream = identity['selected_upstream'][dataset]
    selected = c.read(Path(upstream['folder']) / 'selected_validation.json')
    saved = c.load_upstream_checkpoint(upstream, identity)
    model = c.u.legacy.AdaptationModel('native_detail')
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False).cuda().to(memory_format=torch.channels_last)
    captured = {}
    handle = model.native_model.up_fine.register_forward_hook(lambda m, a, out: captured.update(feature=out))
    entries = []; started = time.monotonic()
    try:
        for split in ('train', 'val'):
            data = c.CachedFine(dataset, split)
            rows = sorted({0, len(data) - 1}) if smoke else range(len(data))
            ids, attn = model.tokenizer.encode(data.description)
            ids, attn = ids.cuda()[None], attn.cuda()[None]
            for row in rows:
                c.stop_check()
                path = folder / 'shards' / f'{split}_{row:04d}.npz'
                if not path.exists():
                    image = np.array(data.images[row], copy=True)
                    low = np.array(data.low[row], copy=True)
                    ni = torch.from_numpy(image).cuda()[None, None]
                    li = torch.from_numpy(low).cuda()[None, None].repeat(1, 3, 1, 1).to(memory_format=torch.channels_last)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, logits = model(ni, li, ids, attn)
                    prediction = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
                    probabilities = logits.float().softmax(1)[0].cpu().numpy()
                    features = captured.pop('feature').float()[0].cpu().numpy()
                    descriptors, components, descriptor_pred = c.describe(image, probabilities, features)
                    np.testing.assert_array_equal(prediction, descriptor_pred)
                    truth = np.asarray(data.masks[row])
                    counts, outside, frozen = s.targets(components, prediction, truth)
                    values = dict(**descriptors, row=np.array(row), split=np.array(split),
                                  label_id=np.array(data.entry['records'][row]['label_id']),
                                  identity_sha256=np.array(ident_hash), prediction=prediction,
                                  components=components.astype(np.int32))
                    for grid in c.GRIDS:
                        values[grid + '_counts'] = counts[grid]
                        values[grid + '_outside'] = outside[grid]
                        values[grid + '_frozen'] = frozen[grid]
                    # Full dense equivalence before the shard is committed.
                    validate_shard(values, row, data, ident_hash)
                    c.save_npz(path, values)
                    del ni, li, logits, features, probabilities, descriptors, values
                with np.load(path, allow_pickle=False) as z:
                    n = validate_shard(z, row, data, ident_hash)
                    if split == 'val':
                        reference = selected['saved_predictions'][row]
                        assert reference['index'] == row and reference['label_id'] == data.entry['records'][row]['label_id']
                        prior_path = Path(upstream['folder']) / selected['predictions_directory'] / reference['file']
                        assert c.sha(prior_path) == reference['sha256']
                        with np.load(prior_path, allow_pickle=False) as prior:
                            np.testing.assert_array_equal(prior['target_native'], data.masks[row])
                            np.testing.assert_array_equal(prior['native_eightclass'], z['prediction'])
                            np.testing.assert_array_equal(prior['canonical512_eightclass'], s.nearest512_ids(z['prediction']))
                entries.append(dict(file=path.name, row=row, split=split,
                                    label_id=data.entry['records'][row]['label_id'],
                                    bodies=n, sha256=c.sha(path)))
                if len(entries) % 25 == 0 or row == rows[-1]:
                    meta.update(completed_images=len(entries), seconds=time.monotonic() - started, updated_utc=c.utc())
                    c.write(folder / 'manifest.json', meta)
                    c.write(c.HERE / 'progress.json', dict(stage='cache', dataset=dataset,
                            split=split, row=row, smoke=smoke, updated_utc=c.utc()))
                    print(f'BODY_CACHE {dataset} {split} row={row} images={len(entries)}', flush=True)
    finally:
        handle.remove(); captured.clear(); model.cpu(); del model, saved
        torch.cuda.empty_cache()
    c.stop_check()
    n = sum(e['bodies'] for e in entries)
    assert n > 0, 'No predicted oil bodies: no head training possible'
    arrays = {k: np.lib.format.open_memmap(folder / (k + '.npy'), mode='w+', dtype=d, shape=(n, *shape))
              for k, (d, shape) in c.LAYOUT.items()}
    outside = {split: {g: np.zeros((8, 8), np.int64) for g in c.GRIDS} for split in ('train', 'val')}
    frozen = {split: {g: np.zeros((8, 8), np.int64) for g in c.GRIDS} for split in ('train', 'val')}
    start = 0
    for entry in entries:
        c.stop_check(); end = start + entry['bodies']; entry.update(start=start, end=end)
        with np.load(folder / 'shards' / entry['file'], allow_pickle=False) as z:
            for key, array in arrays.items():
                array[start:end] = z[key]
            for grid in c.GRIDS:
                outside[entry['split']][grid] += z[grid + '_outside']
                frozen[entry['split']][grid] += z[grid + '_frozen']
        start = end
    for array in arrays.values():
        array.flush()
    del array, arrays
    if not smoke:
        for grid in c.GRIDS:
            np.testing.assert_array_equal(frozen['val'][grid], selected['metrics'][grid]['eightclass']['confusion_matrix'])
    c.write(folder / 'records.json', entries)
    c.write(folder / 'matrices.json', {kind: {split: {g: h.tolist() for g, h in value.items()} for split, value in groups.items()}
                                      for kind, groups in (('outside', outside), ('frozen', frozen))})
    meta.update(status='completed', n=n, completed_utc=c.utc(),
                images_by_split={sp: sum(e['split'] == sp for e in entries) for sp in ('train', 'val')},
                bodies_by_split={sp: sum(e['bodies'] for e in entries if e['split'] == sp) for sp in ('train', 'val')},
                array_sha256={k: c.sha(folder / (k + '.npy')) for k in c.LAYOUT},
                records_sha256=c.sha(folder / 'records.json'), matrices_sha256=c.sha(folder / 'matrices.json'),
                dense_sparse_exact=True, selected_validation_pixel_replay_exact=True)
    c.write(folder / 'manifest.json', meta)
    return verify_cache(folder, identity)


def verify_cache(folder, identity):
    meta = c.read(folder / 'manifest.json')
    assert meta['status'] == 'completed' and meta['identity'] == identity
    assert meta['dense_sparse_exact'] and meta['selected_validation_pixel_replay_exact']
    for name in ('records', 'matrices'):
        assert c.sha(folder / (name + '.json')) == meta[name + '_sha256']
    for key, (dtype, shape) in c.LAYOUT.items():
        values = np.load(folder / (key + '.npy'), mmap_mode='r')
        assert values.dtype == dtype and values.shape == (meta['n'], *shape)
        assert c.sha(folder / (key + '.npy')) == meta['array_sha256'][key]
    for entry in c.read(folder / 'records.json'):
        assert c.sha(folder / 'shards' / entry['file']) == entry['sha256']
    return meta


class Bodies:
    def __init__(self, folder, split):
        if split not in ('train', 'val'):
            raise ValueError('Only TRAIN/VAL; no TEST loader')
        self.folder, self.split = Path(folder), split
        self.meta = c.read(self.folder / 'manifest.json')
        assert self.meta['status'] == 'completed'
        self.records = [e for e in c.read(self.folder / 'records.json') if e['split'] == split]
        self.indices = np.concatenate([np.arange(e['start'], e['end'], dtype=np.int64) for e in self.records])
        self.arrays = {k: np.load(self.folder / (k + '.npy'), mmap_mode='r') for k in c.LAYOUT}
        matrices = c.read(self.folder / 'matrices.json')
        self.outside, self.frozen = matrices['outside'][split], matrices['frozen'][split]

    def inputs(self, indices, device='cuda'):
        return tuple(torch.from_numpy(np.array(self.arrays[k][indices], dtype=d)).to(device)
                     for k, d in (('points', np.float32), ('valid', bool), ('summary', np.float32)))

    def targets(self, indices, device='cuda'):
        return torch.from_numpy(np.array(self.arrays['native_counts'][indices], np.float32)).to(device)


@torch.inference_mode()
def evaluate(model, data, device='cuda', save_to=None):
    if model is not None:
        model.eval()
    matrices = {g: np.asarray(h, np.int64).copy() for g, h in data.outside.items()}
    assignments = []
    for start in range(0, len(data.indices), 512):
        c.stop_check(); indices = data.indices[start:start + 512]
        chosen = (data.arrays['summary'][indices, 3:8].argmax(1) if model is None else
                  model(*data.inputs(indices, device)).argmax(1).cpu().numpy())
        assignments.append(np.asarray(chosen, np.uint8))
        for grid in c.GRIDS:
            matrices[grid] = c.aggregate(data.arrays[grid + '_counts'][indices], chosen, matrices[grid])
    assigned = np.concatenate(assignments) if assignments else np.empty(0, np.uint8)
    for grid in c.GRIDS:
        replay = s.recount(assigned, data.arrays[grid + '_counts'][data.indices], data.outside[grid])
        np.testing.assert_array_equal(replay, matrices[grid])
    s.assert_fixed_support(matrices, data.frozen)
    if save_to is not None:
        c.save_npz(save_to, dict(indices=data.indices, assigned=assigned))
    return {g: c.u.metrics(h) for g, h in matrices.items()}
