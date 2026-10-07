"""Reuse established models/training mechanics in memory; never edit old runs."""
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parent
OLD=EXP/'external_clean_prior_adaptation_20261001'
LABELS=EXP/'external_visual_annotation_20261003/fullset_assisted_v7'
sys.path.insert(0,str(OLD))
import run_adaptation as legacy
import prior_schema
c=legacy.c
read,write,sha=c.read,c.write,c.sha
OLD_CACHE=c.PREVIOUS/'cache'
ORIGINAL_VERIFY=legacy.verify_cache
BASE_CONFIG=dict(c.CONFIG)
c.HERE=HERE
c.CONFIG={**BASE_CONFIG,'protocol':'external_fine8_v7_stage1_seed42_v1',
          'primary_rule':'eightclass','label_space':'fine8','label_version':'external_3750_v7',
          'model_test_pixels_read':False,'annotation_test_blocks_previously_reviewed':True}
c.RULES=('eightclass',)

def stop_check():
    # Previous experiments' markers remain in place and do not authorize restarting them.
    for folder in (HERE,ROOT):
        for name in ('STOP.txt','STOP_AFTER_EPOCH.txt'):
            if (folder/name).exists():raise InterruptedError(f'User stop marker: {folder/name}')
c.stop_check=stop_check

def fine_weights(counts):
    x=np.asarray(counts,dtype=np.float64)
    assert x.shape==(8,) and (x>=0).all() and x.sum()>0
    w=1/np.sqrt(x/x.sum()+1e-8)
    return np.clip(w/w.mean(),.3,6).astype(np.float32)

def fine_loss(logits,target,weights):
    if logits.ndim!=4 or logits.shape[1]!=8 or weights.shape!=(8,):raise ValueError('fine8 loss contract')
    if target.dtype!=torch.long or target.min()<0 or target.max()>7:raise ValueError('fine8 target contract')
    logp=F.log_softmax(logits.float(),dim=1)
    return .5*F.nll_loss(logp,target,weight=weights)+.5*c.weighted_dice_from_logp(logp,target,weights)

def histogram(pred,target):
    if pred.shape!=target.shape or min(pred.min(),target.min())<0 or max(pred.max(),target.max())>7:raise ValueError('Aligned fine8 maps required')
    return np.bincount((target.astype(np.int64)*8+pred).ravel(),minlength=64).reshape(8,8)

def metrics(h):
    h=np.asarray(h,np.int64);assert h.shape==(8,8)
    diag=np.diag(h).astype(np.float64);gt=h.sum(1);pred=h.sum(0)
    iou=diag/(gt+pred-diag+1e-8);dice=2*diag/(gt+pred+1e-8)
    return {'mIoU':float(iou.mean()),'mDice':float(dice.mean()),'oil_macro_IoU':float(iou[3:].mean()),
            'nonbackground_mIoU':float(iou[1:].mean()),'class_IoU':iou.tolist(),'class_Dice':dice.tolist(),
            'gt_pixels':gt.tolist(),'pred_pixels':pred.tolist(),'confusion_matrix':h.tolist(),
            'absent_class_convention':'fixed 8-class mean; empty union yields zero'}

def identity(include_cache=True):
    parent=read(OLD/'run_manifest.json');assert parent['status']=='completed'
    sources=dict(parent['identity']['sources']);inputs=dict(parent['identity']['inputs'])
    for entries in (sources,inputs):
        for path,digest in entries.items():assert sha(path)==digest,path
    for name in ('bridge.py','data8.py','run_eightclass.py','test_eightclass.py','PROTOCOL.md'):
        sources[str(HERE/name)]=sha(HERE/name)
    for path in (HERE/'expert_acceptance.json',HERE/'label_freeze.json',HERE/'cache/manifest.json'):
        inputs[str(path)]=sha(path)
    return {'sources':sources,'inputs':inputs,'config':c.CONFIG,
            'label_fingerprint':read(HERE/'label_freeze.json')['dataset_fingerprint'],
            'cache_manifest':sha(HERE/'cache/manifest.json')}
c.identity=identity
