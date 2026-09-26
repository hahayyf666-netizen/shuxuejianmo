"""Read-only availability probe; transcript IDs are never promoted to lineage."""
import concurrent.futures
import json
import re
from collections import Counter
from pathlib import Path

import requests

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results/source_candidates_20.json').read_text(encoding='utf-8'))

def probe(item):
    sid=item['sample_id']
    top=item['top_candidates'][0] if item['top_candidates'] else None
    row={'sample_id':sid,'candidate_video_id':top['video_id'] if top else None,
         'trigram_hits':top['trigram_hits'] if top else 0,
         'ordered_exact_tokens':top['ordered_exact_tokens'] if top else 0,
         'target_tokens':top['target_tokens'] if top else None,
         'identity_status':'transcript_candidate_only_not_feature_verified'}
    vid=row['candidate_video_id']
    if not vid or not re.fullmatch(r'[A-Za-z0-9_-]{11}',vid):
        row.update({'oembed_status':'NOT_YOUTUBE_ID_SHAPE','http_status':None,'title':None})
        return row
    try:
        resp=requests.get('https://www.youtube.com/oembed',params={'url':f'https://www.youtube.com/watch?v={vid}','format':'json'},timeout=15)
        title=resp.json().get('title') if resp.status_code==200 else None
        row.update({'oembed_status':'AVAILABLE_METADATA' if resp.status_code==200 else 'UNAVAILABLE_AT_PROBE',
                    'http_status':resp.status_code,'title':title})
    except Exception as exc:
        row.update({'oembed_status':'NETWORK_ERROR','http_status':None,'title':None,'error':type(exc).__name__})
    return row

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(probe,data['samples']))
out={'rule':'oEmbed metadata only; candidate ID from transcript does not verify feature provenance',
     'counts':dict(Counter(r['oembed_status'] for r in rows)),'samples':rows}
(HERE/'results/source_availability_20.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out['counts']))
print(json.dumps([(r['sample_id'],r['candidate_video_id'],r['oembed_status']) for r in rows]))
