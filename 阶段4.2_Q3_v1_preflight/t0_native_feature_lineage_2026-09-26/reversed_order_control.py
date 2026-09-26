"""Execute the second predeclared T0 negative control on normal samples 02/03.

Reverse the *actual candidate feature array*, repeat nearest-neighbor matching,
then apply the complete pass gate including chronological row order. Matching
values should remain exact, while source indices must become decreasing.
"""
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from remote_h5 import HTTPRangeFile

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'results/t0_frozen_contract.json').read_text(encoding='utf-8'))
records = {x['sample_id']: x for x in json.loads((HERE / 'results/t0_transcript_source_candidates.json').read_text(encoding='utf-8'))['samples']}
out = {'control': 'reversed_candidate_order', 'samples': [], 'all_full_gates_reject': True}
for sid in ('02', '03'):
    frozen = next(x for x in contract['sources'] if x['sample_id'] == sid)
    source_video_id = records[sid]['top_candidates'][0]['source_video_id']
    with open(frozen['files']['unaligned_pkl']['path'], 'rb') as h:
        official = pickle.load(h)
    item = {'sample_id': sid, 'source_video_id': source_video_id, 'modalities': {}}
    for modality, url_key in (('audio', 'covarep_url'), ('vision', 'facet_url')):
        remote = HTTPRangeFile(contract['candidate_source'][url_key])
        with h5py.File(remote, 'r') as h:
            root = next(iter(h.keys()))
            candidate = np.asarray(h[root]['data'][source_video_id]['features'][:], dtype=np.float64)
        candidate = candidate[::-1].copy()
        target = np.asarray(official[modality][:int(official[modality+'_lengths'])], dtype=np.float64)
        finite_idx = np.flatnonzero(np.isfinite(candidate).all(axis=1))
        distances, local_idx = cKDTree(candidate[finite_idx]).query(target, k=2)
        idx = finite_idx[local_idx]
        nearest = candidate[idx[:, 0]]
        maxabs = np.max(np.abs(target - nearest), axis=1)
        rel = distances[:, 0] / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
        second_maxabs = np.max(np.abs(target - candidate[idx[:, 1]]), axis=1)
        strict = (maxabs <= 1e-4) & (rel <= 1e-6) & (second_maxabs >= 1e-3)
        matched_idx = idx[strict, 0]
        chronological = bool(len(matched_idx)==len(target) and np.all(np.diff(matched_idx)>0))
        reversed_order = bool(len(matched_idx)==len(target) and np.all(np.diff(matched_idx)<0))
        full_gate = bool(strict.all() and chronological)
        rec = {'official_rows': len(target), 'strict_value_matches': int(strict.sum()),
               'nearest_maxabs_max': float(maxabs.max()), 'reversed_indices': reversed_order,
               'chronological_indices': chronological, 'first_indices': idx[:3, 0].tolist(),
               'last_indices': idx[-3:, 0].tolist(), 'full_gate_pass': full_gate,
               'network_bytes': remote.bytes_transferred, 'candidate_etag': remote.etag}
        item['modalities'][modality] = rec
        out['all_full_gates_reject'] &= not full_gate
        print(sid, modality, int(strict.sum()), '/', len(target), 'chronological=', chronological,
              'reversed=', reversed_order, 'full_gate=', full_gate, flush=True)
    out['samples'].append(item)
(HERE / 'results/t0_reversed_order_negative_control.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
assert out['all_full_gates_reject']
