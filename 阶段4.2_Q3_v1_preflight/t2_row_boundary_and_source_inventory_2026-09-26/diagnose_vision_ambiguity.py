"""Read-only diagnosis of the two non-unique zero-error FACET rows."""
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
source_doc=json.loads((HERE/'results/native20_frozen_rules.json').read_text(encoding='utf-8'))
contract=json.loads((T0/'results/t0_frozen_contract.public.json').read_text(encoding='utf-8'))
remote=HTTPRangeFile(contract['candidate_source']['facet_url'])
base=ROOT/'E题数据/附件4-可解释专项视频样本与特征文件/未对齐版本'
out=[]
with h5py.File(remote,'r') as h:
    data=h[next(iter(h.keys()))]['data']
    for sid,j in [('06',95),('18',73)]:
        vid=source_doc['sample_to_candidate'][sid]
        with (base/f'{sid}.pkl').open('rb') as handle: official=pickle.load(handle)
        target=np.asarray(official['vision'][j],dtype=np.float64)
        g=data[vid]
        features=np.asarray(g['features'][:],dtype=np.float64)
        intervals=np.asarray(g['intervals'][:],dtype=np.float64)
        finite_idx=np.flatnonzero(np.isfinite(features).all(axis=1))
        distances,idx=cKDTree(features[finite_idx]).query(target,k=5)
        idx=finite_idx[idx]
        candidates=[{'native_r':int(r),'maxabs':float(np.max(np.abs(features[r]-target))),
                     'l2':float(d),'source_time':intervals[r].tolist()} for r,d in zip(idx,distances)]
        out.append({'sample_id':sid,'official_unaligned_j':j,'candidate_video_id':vid,
                    'first_five_nearest_source_rows':candidates,
                    'decision':'nonunique_source_row_no_timestamp_assignment'})
        print(sid,candidates[:2],flush=True)
(HERE/'results/vision_ambiguity_06_18.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
