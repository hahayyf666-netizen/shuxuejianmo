"""Exact COVAREP/FACET source-row lineage for 20 Attachment4 samples.

Reuses the five frozen T0 result files; checks the remaining 15 top transcript
candidate groups against official unaligned rows. It never assigns clip times.
"""
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
T0=HERE.parent/'t0_native_feature_lineage_2026-09-26'
sys.path.insert(0,str(T0))
from remote_h5 import HTTPRangeFile

freeze=json.loads((HERE/'results/native20_frozen_rules.json').read_text(encoding='utf-8'))
t0_contract=json.loads((T0/'results/t0_frozen_contract.public.json').read_text(encoding='utf-8'))
base=ROOT/'E题数据/附件4-可解释专项视频样本与特征文件/未对齐版本'
all_results={}
for sid in ('02','03','07','13','16'):
    old=json.loads((T0/'results'/f't0_native_row_match_{sid}.json').read_text(encoding='utf-8'))
    all_results[sid]={m:{'official_rows':old['modalities'][m]['official_shape'][0],
                         'strict_unique_matches':old['modalities'][m]['strict_unique_matches'],
                         'source_rows_strictly_increasing':old['modalities'][m]['strict_ordered'],
                         'all_zero_maxabs':old['modalities'][m]['nearest_maxabs_min_median_max'][2]==0,
                         'source_row_count':old['modalities'][m]['native_shape'][0],
                         'strict_rows':old['modalities'][m]['strict_rows'],
                         'source_kind':'T0_reused'} for m in ('audio','vision')}
# The partial file below is a crash diagnostic only. Every authoritative run
# recomputes the 15 new samples and reuses only the published T0 five.

for modality,url_key in (('audio','covarep_url'),('vision','facet_url')):
    remote=HTTPRangeFile(t0_contract['candidate_source'][url_key],block_size=1024*1024,max_blocks=128)
    with h5py.File(remote,'r') as h:
        data=h[next(iter(h.keys()))]['data']
        for i in range(1,21):
            sid=f'{i:02d}'
            if modality in all_results.get(sid,{}): continue
            vid=freeze['sample_to_candidate'][sid]
            if not vid or vid not in data:
                all_results.setdefault(sid,{})[modality]={'status':'CANDIDATE_SOURCE_GROUP_MISSING'}
                print(sid,modality,'MISSING',flush=True); continue
            with (base/f'{sid}.pkl').open('rb') as handle: official=pickle.load(handle)
            count=int(official[modality+'_lengths'])
            target=np.asarray(official[modality][:count],dtype=np.float64)
            g=data[vid]
            source=np.asarray(g['features'][:],dtype=np.float64)
            intervals=np.asarray(g['intervals'][:],dtype=np.float64)
            if source.ndim!=2 or source.shape[1]!=target.shape[1] or len(intervals)!=len(source):
                all_results.setdefault(sid,{})[modality]={'status':'DIMENSION_OR_INTERVAL_SHAPE_MISMATCH'}
                print(sid,modality,'DIMENSION_MISMATCH',flush=True); continue
            if not np.isfinite(target).all():
                all_results.setdefault(sid,{})[modality]={'status':'OFFICIAL_NONFINITE'}
                print(sid,modality,'OFFICIAL_NONFINITE',flush=True); continue
            finite_idx=np.flatnonzero(np.isfinite(source).all(axis=1))
            if len(finite_idx)<2:
                all_results.setdefault(sid,{})[modality]={'status':'INSUFFICIENT_FINITE_SOURCE'}
                print(sid,modality,'INSUFFICIENT_FINITE_SOURCE',flush=True); continue
            d,local_idx=cKDTree(source[finite_idx]).query(target,k=2,workers=1)
            idx=finite_idx[local_idx]
            nearest=source[idx[:,0]]
            maxabs=np.max(np.abs(target-nearest),axis=1)
            rel=d[:,0]/np.maximum(np.linalg.norm(target,axis=1),1e-12)
            second_abs=np.max(np.abs(target-source[idx[:,1]]),axis=1)
            strict=(maxabs<=1e-4)&(rel<=1e-6)&(second_abs>=1e-3)
            strict_indices=idx[strict,0]
            ordered=bool(np.all(np.diff(strict_indices)>0)) if len(strict_indices)>1 else True
            valid_time=bool(np.all(intervals[strict_indices,1]>intervals[strict_indices,0]))
            rowmap=[{'official_j':int(j),'native_r':int(idx[j,0]),
                     'native_time':intervals[idx[j,0]].tolist()} for j in np.flatnonzero(strict)]
            status='PASS' if int(strict.sum())==count and ordered and valid_time else 'FAIL'
            all_results.setdefault(sid,{})[modality]={
                'status':status,'official_rows':count,'strict_unique_matches':int(strict.sum()),
                'source_rows_strictly_increasing':ordered,'positive_source_intervals':valid_time,
                'all_zero_maxabs':bool(np.all(maxabs==0)),
                'nearest_maxabs_min_median_max':[float(maxabs.min()),float(np.median(maxabs)),float(maxabs.max())],
                'source_row_count':len(source),'native_etag':remote.etag,
                'strict_rows':rowmap,'source_kind':'T2_new'}
            print(sid,modality,int(strict.sum()),'/',count,status,flush=True)
            (HERE/'results/native20_row_results.partial.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
    print(modality,'network_bytes',remote.bytes_transferred,flush=True)

summary={'candidate_rule':'top transcript CSD group only, then strict native feature row match',
         'mirror_commit':freeze['source_mirror_commit'],'samples':[]}
for i in range(1,21):
    sid=f'{i:02d}'
    item={'sample_id':sid,'candidate_video_id':freeze['sample_to_candidate'][sid],
          'audio':all_results.get(sid,{}).get('audio',{}),'vision':all_results.get(sid,{}).get('vision',{})}
    summary['samples'].append(item)
(HERE/'results/native20_row_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('DONE',len(summary['samples']))
