"""Strict, read-only row matching of one frozen sample to timestamped native CSD.

The public CSD is a candidate, not assumed provenance. No time or row index
ratio enters candidate selection; exact feature-space nearest rows do.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from remote_h5 import HTTPRangeFile

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'results/t0_frozen_contract.json').read_text(encoding='utf-8'))
ids = json.loads((HERE / 'results/t0_transcript_source_candidates.json').read_text(encoding='utf-8'))
sample = sys.argv[1] if len(sys.argv) > 1 else '02'
frozen = next(s for s in contract['sources'] if s['sample_id'] == sample)
source = next(s['top_candidates'][0] for s in ids['samples'] if s['sample_id'] == sample)
video_id = source['source_video_id']
with open(frozen['files']['unaligned_pkl']['path'], 'rb') as handle:
    official = pickle.load(handle)

out = {'sample_id': sample, 'candidate_video_id': video_id, 'source_phrase': source,
       'candidate_source_commit': contract['candidate_source']['mirror_commit'], 'modalities': {}}
for modality, url_key in [('audio', 'covarep_url'), ('vision', 'facet_url')]:
    file = HTTPRangeFile(contract['candidate_source'][url_key], block_size=1024*1024, max_blocks=128)
    with h5py.File(file, 'r') as h:
        root = next(iter(h.keys()))
        data = h[root]['data']
        if video_id not in data:
            out['modalities'][modality] = {'status': 'CANDIDATE_VIDEO_MISSING'}
            continue
        group = data[video_id]
        rows = np.asarray(group['features'][:], dtype=np.float64)
        intervals = np.asarray(group['intervals'][:], dtype=np.float64)
    count = int(official[modality + '_lengths'])
    target = np.asarray(official[modality][:count], dtype=np.float64)
    assert rows.ndim == target.ndim == 2 and rows.shape[1] == target.shape[1]
    assert intervals.shape == (rows.shape[0], 2)
    native_finite = np.isfinite(rows).all(axis=1)
    target_finite = np.isfinite(target).all(axis=1)
    if not target_finite.all():
        raise ValueError('official unaligned rows contain nonfinite values')
    finite_indices = np.flatnonzero(native_finite)
    if len(finite_indices) < 2:
        raise ValueError('candidate CSD has fewer than two finite rows')
    tree = cKDTree(rows[finite_indices])
    distances, row_indices = tree.query(target, k=2, workers=1)
    row_indices = finite_indices[row_indices]
    first = rows[row_indices[:, 0]]
    maxabs = np.max(np.abs(target - first), axis=1)
    norm = np.linalg.norm(target, axis=1)
    rel_l2 = distances[:, 0] / np.maximum(norm, 1e-12)
    second_maxabs = np.max(np.abs(target - rows[row_indices[:, 1]]), axis=1)
    strict = ((maxabs <= 1e-4) & (rel_l2 <= 1e-6) & (second_maxabs >= 1e-3))
    strict_idx = row_indices[strict, 0]
    output = {
        'status': 'ROW_MATCH_DIAGNOSTIC',
        'official_shape': list(target.shape), 'native_shape': list(rows.shape),
        'native_root': root, 'native_etag': file.etag,
        'native_nonfinite_rows': int((~native_finite).sum()),
        'native_interval_minmax': [float(np.min(intervals[:, 0])), float(np.max(intervals[:, 1]))],
        'strict_unique_matches': int(strict.sum()),
        'strict_coverage': float(strict.mean()),
        'strict_ordered': bool(np.all(np.diff(strict_idx) > 0)) if len(strict_idx) > 1 else None,
        'nearest_maxabs_min_median_max': [float(np.min(maxabs)), float(np.median(maxabs)), float(np.max(maxabs))],
        'nearest_rel_l2_min_median_max': [float(np.min(rel_l2)), float(np.median(rel_l2)), float(np.max(rel_l2))],
        'nearest_second_maxabs_min': float(np.min(second_maxabs)),
        'first_ten': [{'official_j': int(j), 'native_r': int(row_indices[j, 0]),
                       'maxabs': float(maxabs[j]), 'rel_l2': float(rel_l2[j]),
                       'native_time': intervals[row_indices[j, 0]].tolist(),
                       'strict': bool(strict[j])} for j in range(min(10, count))],
        'strict_rows': [{'official_j': int(j), 'native_r': int(row_indices[j, 0]),
                         'native_time': intervals[row_indices[j, 0]].tolist()}
                        for j in np.flatnonzero(strict)],
        'network_requests': file.requests_made, 'network_bytes': file.bytes_transferred,
    }
    out['modalities'][modality] = output
    print(modality, 'strict', output['strict_unique_matches'], '/', count,
          'min maxabs', output['nearest_maxabs_min_median_max'][0],
          'median maxabs', output['nearest_maxabs_min_median_max'][1], flush=True)

path = HERE / 'results' / f't0_native_row_match_{sample}.json'
path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print('saved', path, flush=True)
