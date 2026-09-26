"""Frozen wrong-video negative control: official 02 vs native video for 03."""
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from remote_h5 import HTTPRangeFile

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'results/t0_frozen_contract.json').read_text(encoding='utf-8'))
sources = json.loads((HERE / 'results/t0_transcript_source_candidates.json').read_text(encoding='utf-8'))['samples']
wrong_video = next(s['top_candidates'][0]['source_video_id'] for s in sources if s['sample_id'] == '03')
official_path = next(s['files']['unaligned_pkl']['path'] for s in contract['sources'] if s['sample_id'] == '02')
with open(official_path, 'rb') as h:
    official = pickle.load(h)
results = {'sample_id': '02', 'wrong_video_sample_id': '03', 'wrong_video_id': wrong_video, 'modalities': {}}
for modality, key in [('audio', 'covarep_url'), ('vision', 'facet_url')]:
    remote = HTTPRangeFile(contract['candidate_source'][key])
    with h5py.File(remote, 'r') as h:
        rows = np.asarray(h[next(iter(h.keys()))]['data'][wrong_video]['features'][:], dtype=np.float64)
    rows = rows[np.isfinite(rows).all(axis=1)]
    target = np.asarray(official[modality][:int(official[modality + '_lengths'])], dtype=np.float64)
    d, idx = cKDTree(rows).query(target)
    maxabs = np.max(np.abs(target - rows[idx]), axis=1)
    strict = (maxabs <= 1e-4) & (d / np.maximum(np.linalg.norm(target, axis=1), 1e-12) <= 1e-6)
    results['modalities'][modality] = {'strict_matches': int(strict.sum()), 'target_rows': len(target),
                                      'minimum_maxabs': float(np.min(maxabs)), 'network_bytes': remote.bytes_transferred}
    print(modality, results['modalities'][modality], flush=True)
(HERE / 'results/t0_wrong_video_negative_control.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
