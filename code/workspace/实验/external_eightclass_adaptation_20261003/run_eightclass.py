"""Stage-one eight-class adaptation; existing model and loop, new fixed labels."""
import argparse
import importlib.util
import os
import sys
import time
import unittest
import numpy as np
import torch
import bridge as b
from data8 import CachedFine,verify

def counts():return {g:{'eightclass':np.zeros((8,8),np.int64)} for g in b.c.GRIDS}

def primary(report):return report['metrics']['canonical512']['eightclass']['mIoU']

@torch.inference_mode()
def evaluate(model,dataset,save_to=None):
    model.eval();ids,attention=b.legacy.tokens(model,dataset);matrices=counts();saved=[];start=time.monotonic()
    if save_to is not None:save_to.mkdir(exist_ok=False)
    for i in range(len(dataset)):
        b.stop_check();sample=dataset.get(i);image,low,target_gpu=b.legacy.to_device(sample)
        with torch.autocast('cuda',dtype=torch.bfloat16):coarse,native=model(image,low,ids,attention)
        pred_native=native.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred512=(coarse.float().argmax(1)[0].cpu().numpy().astype(np.uint8) if model.arm=='global'
                 else b.c.data.nearest512(pred_native))
        predictions={'native':pred_native,'canonical512':pred512}
        target=sample[2]
        for grid in b.c.GRIDS:
            truth=target if grid=='native' else b.c.data.nearest512(target)
            matrices[grid]['eightclass']+=b.histogram(predictions[grid],truth)
        if save_to is not None:
            path=save_to/f'{i:04d}.npz';tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,target_native=target,**{g+'_eightclass':p for g,p in predictions.items()})
            tmp.replace(path);saved.append({'index':i,'label_id':dataset.entry['records'][i]['label_id'],
                                           'file':path.name,'sha256':b.sha(path)})
        del image,low,target_gpu,coarse,native
    result={'images':len(dataset),'split':dataset.split,'label_space':'fine8',
            'metrics':{g:{'eightclass':b.metrics(matrices[g]['eightclass'])} for g in b.c.GRIDS},
            'seconds':time.monotonic()-start,'external_test_pixels_read':False}
    if save_to is not None:
        independent=counts()
        for row in saved:
            path=save_to/row['file'];assert b.sha(path)==row['sha256']
            with np.load(path,allow_pickle=False) as arr:
                for grid in b.c.GRIDS:
                    truth=arr['target_native'] if grid=='native' else b.c.data.nearest512(arr['target_native'])
                    pred=arr[grid+'_eightclass']
                    # Reverse encoding and transpose, independent of the live histogram function.
                    h=np.bincount((pred.astype(np.int64)*8+truth).ravel(),minlength=64).reshape(8,8).T
                    independent[grid]['eightclass']+=h
        for grid in b.c.GRIDS:np.testing.assert_array_equal(independent[grid]['eightclass'],matrices[grid]['eightclass'])
        result.update(saved_predictions=saved,saved_pixel_replay_exact=True)
    return result

def summary(reports):
    result={'status':'completed','stage':'upstream_adaptation_only','seed':42,'datasets':{},
            'label_space':'fine8','primary_grid':'canonical512','primary_rule':'eightclass',
            'replication_screen_passed':None,'complete_body_model_evaluated':False,
            'external_test_pixels_read':False,'main_model_promoted':False}
    lines=['# 外部八类适配：上游支撑网络 seed42','',
           '验证集成绩；非测试、非零样本，油体形态头尚未接入。本表不得当作完整主模型结论。','',
           '| 数据集 | Global 八类mIoU | Native 八类mIoU | 差值/百分点 | 最佳轮 G/N |',
           '|---|---:|---:|---:|---|']
    for dataset in b.c.DATASETS:
        a=reports[dataset+'_global'];n=reports[dataset+'_native_detail']
        differences={g:{'mIoU_pp':100*(n['metrics'][g]['eightclass']['mIoU']-a['metrics'][g]['eightclass']['mIoU']),
                        'oil_macro_IoU_pp':100*(n['metrics'][g]['eightclass']['oil_macro_IoU']-a['metrics'][g]['eightclass']['oil_macro_IoU']),
                        'class_IoU_pp':(100*(np.asarray(n['metrics'][g]['eightclass']['class_IoU'])-a['metrics'][g]['eightclass']['class_IoU'])).tolist()}
                     for g in b.c.GRIDS}
        result['datasets'][dataset]={'global':a,'native_detail':n,'differences':differences}
        lines.append(f"| {dataset} | {primary(a):.6f} | {primary(n):.6f} | {100*(primary(n)-primary(a)):+.4f} | {a['selected_epoch']}/{n['selected_epoch']} |")
    lines+=['','全部逐类及双网格结果见 summary.json。旧四相标签和指标不与本表直接混算。',
            '无论胜负均保留本轮，后续同上游的油体判别对照用于完整链条验证；不据此反向修改标签。','']
    b.write(b.HERE/'summary.json',result);(b.HERE/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

def install():
    # Bind only the in-process imported functions; the original files remain untouched.
    b.legacy.phase_loss=b.fine_loss;b.legacy.Cached=CachedFine;b.legacy.verify_cache=verify
    b.legacy.primary=primary;b.legacy.counts=counts;b.legacy.evaluate=evaluate;b.legacy.summary=summary

def contracts():
    spec=importlib.util.spec_from_file_location('external_fine8_contracts',b.HERE/'test_eightclass.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert result.wasSuccessful();return result.testsRun

def smoke():
    b.stop_check();test_count=contracts();verify();identity=b.identity();b.c.setup();reports={}
    for dataset in b.c.DATASETS:
        data=CachedFine(dataset,'train');initial_global=None
        for arm in b.c.ARMS:
            model=b.legacy.AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
            parity=b.legacy.legacy_id_gpu_parity(model,data)
            optimizer=b.legacy.optimizer_for(model)
            start_global=b.c.state_hash(model.global_model.state_dict())
            if initial_global is None:initial_global=start_global
            assert start_global==initial_global
            native_start=b.c.state_hash(model.native_model.state_dict()) if arm=='native_detail' else None
            rows=model.global_model.text_encoder.token_embed.new_weight.detach().clone()
            torch.cuda.reset_peak_memory_stats()
            train=b.legacy.train_epoch(model,data,optimizer,1,plan=[(0,0),(len(data)-1,1),(0,4),(len(data)-1,8)])
            assert train['images']==4 and train['updates']==1
            assert b.c.state_hash(model.global_model.state_dict())!=start_global
            assert not torch.equal(rows,model.global_model.text_encoder.token_embed.new_weight)
            if arm=='native_detail':assert b.c.state_hash(model.native_model.state_dict())!=native_start
            reports[dataset+'_'+arm]={'original_token_inference_parity':parity,'initial_global_hash':start_global,
                'training':train,'parameters':sum(p.numel() for p in model.parameters()),
                'trainable_parameters':sum(p.numel() for p in model.trainable_parameters()),
                'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'smoke_weights_discarded':True}
            b.legacy.close_model(model,optimizer);del model,optimizer
    assert b.identity()==identity
    b.write(b.HERE/'smoke_report.json',{'status':'passed','cpu_tests':test_count,'identity':identity,
            'reports':reports,'external_test_pixels_read':False,'completed_utc':b.c.utc()})
    print('ALL_FOUR_FINE8_SMOKES_PASS',flush=True)

def verify_completed():
    identity=b.identity();meta=b.read(b.HERE/'run_manifest.json');assert meta['identity']==identity and meta['status']=='completed'
    reports=[]
    for dataset in b.c.DATASETS:
        for arm in b.c.ARMS:
            folder=b.legacy.arm_folder(dataset,arm);m=b.read(folder/'run_manifest.json');h=m['history']
            assert m['identity']==identity and m['status']=='completed' and len(h)==13
            assert [r['epoch'] for r in h]==list(range(13))
            assert m['best_epoch']==max(range(13),key=lambda j:primary(h[j]['validation']))
            for e in m['epochs']:assert b.sha(folder/'epochs'/e['file'])==e['sha256']
            selected=b.read(folder/'selected_validation.json');assert selected['saved_pixel_replay_exact']
            assert b.sha(folder/'selected_validation.json')==m['selected_sha256']
            mats=counts();data=CachedFine(dataset,'val')
            for row in selected['saved_predictions']:
                path=folder/selected['predictions_directory']/row['file'];assert b.sha(path)==row['sha256']
                with np.load(path,allow_pickle=False) as arr:
                    np.testing.assert_array_equal(arr['target_native'],data.masks[row['index']])
                    for grid in b.c.GRIDS:
                        target=arr['target_native'] if grid=='native' else b.c.data.nearest512(arr['target_native'])
                        mats[grid]['eightclass']+=b.histogram(arr[grid+'_eightclass'],target)
            for grid in b.c.GRIDS:
                np.testing.assert_array_equal(mats[grid]['eightclass'],selected['metrics'][grid]['eightclass']['confusion_matrix'])
                assert b.metrics(mats[grid]['eightclass'])==selected['metrics'][grid]['eightclass']
            reports.append({'dataset':dataset,'arm':arm,'best_epoch':m['best_epoch'],'predictions':len(data),'checks':'PASS'})
    b.write(b.HERE/'verification.json',{'status':'PASS','reports':reports,'histories':52,'saved_prediction_records':780,
            'label_space':'fine8','external_test_pixels_read':False,'verified_utc':b.c.utc()})
    print('FINE8_COMPLETED_VERIFIED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['contracts','smoke','run','verify'])
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();install()
    with b.c.lock():
        if args.stage=='run':b.legacy.run(args.resume);verify_completed()
        elif args.stage=='verify':verify_completed()
        else:globals()[args.stage]()
