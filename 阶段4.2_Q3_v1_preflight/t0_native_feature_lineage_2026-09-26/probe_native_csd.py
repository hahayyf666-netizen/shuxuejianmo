"""Read only one identified source video group from each pinned CSD."""
import json
import sys
from pathlib import Path

import h5py

from remote_h5 import HTTPRangeFile

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'results/t0_frozen_contract.json').read_text(encoding='utf-8'))
ids = json.loads((HERE / 'results/t0_transcript_source_candidates.json').read_text(encoding='utf-8'))
sample = sys.argv[1] if len(sys.argv) > 1 else '02'
video_id = next(x['top_candidates'][0]['source_video_id'] for x in ids['samples'] if x['sample_id'] == sample)
for modality, url_key in [('audio', 'covarep_url'), ('vision', 'facet_url')]:
    url = contract['candidate_source'][url_key]
    f = HTTPRangeFile(url, block_size=1024*1024, max_blocks=64)
    print('OPEN', modality, 'bytes', f.size, 'etag', f.etag, flush=True)
    with h5py.File(f, 'r') as h:
        root = list(h.keys())
        print('ROOT', root, flush=True)
        data = h[root[0]]['data']
        print('VIDEO ID', video_id, 'present', video_id in data, flush=True)
        if video_id in data:
            g = data[video_id]
            print('GROUP', list(g.keys()), flush=True)
            for name in g:
                d = g[name]
                print('DATASET', name, d.shape, d.dtype, 'chunks', d.chunks, flush=True)
                print('FIRST', d[0].tolist(), flush=True)
    print('NETWORK', f.requests_made, f.bytes_transferred, flush=True)
