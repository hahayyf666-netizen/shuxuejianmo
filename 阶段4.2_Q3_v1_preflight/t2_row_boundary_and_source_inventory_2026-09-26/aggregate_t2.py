"""Aggregate C2 + native CSD + availability without altering formal XAI."""
import csv
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
native=json.loads((HERE/'results/native20_row_results.json').read_text(encoding='utf-8'))
availability=json.loads((HERE/'results/source_availability_20.json').read_text(encoding='utf-8'))
rowmap={s['sample_id']:s for s in native['samples']}
available={s['sample_id']:s for s in availability['samples']}
c2_path=ROOT/'阶段4.2_Q3_v1_preflight/stage_c2_provenance/results/aligned_to_unaligned_row_matches_20.csv'
with c2_path.open(encoding='utf-8-sig',newline='') as handle:
    c2=list(csv.DictReader(handle))
navigation_path=HERE/'results/sample02_row_navigation.csv'
with navigation_path.open(encoding='utf-8',newline='') as handle:
    navigation={(r['modality'],r['official_seq_index']):r for r in csv.DictReader(handle)}
rows=[]
for r in c2:
    sid,modality=r['sample_id'],r['modality']
    seq=r['official_seq_index']
    n=rowmap[sid][modality]
    indices=[int(x) for x in r['exact_unaligned_indices_zero_based'].split(';') if x]
    c2_ok=len(indices)==1 and int(r['exact_source_index_count'])==1
    native_indices={x['official_j'] for x in n['strict_rows']}
    native_ok=c2_ok and indices[0] in native_indices
    nav=navigation.get((modality,seq)) if sid=='02' else None
    media_origin='PASS_MEDIA_02' if sid=='02' else 'UNVERIFIED'
    status='SOURCE_ROW_UNIQUE' if native_ok else ('C2_CHAIN_UNAVAILABLE' if not c2_ok else 'NATIVE_ROW_NONUNIQUE')
    rows.append({'sample_id':sid,'modality':modality,'official_seq_index':seq,
                 'unaligned_j':indices[0] if c2_ok else '',
                 'c2_unique':int(c2_ok),'native_unique':int(native_ok),
                 'source_row_status':status,'media_origin_status':media_origin,
                 'local_navigation_candidate':int(bool(nav and nav['mapping_status']=='reconstructed_media_navigation_candidate_only')),
                 'formal_mapping_status':'index_only'})
path=HERE/'results/aligned_position_lineage_20.csv'
with path.open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
out={'attachment4_sample_count':20,'aligned_position_count':len(rows),
     'native_unaligned_rows':{m:{'matched':sum(s[m]['strict_unique_matches'] for s in native['samples']),
                                 'total':sum(s[m]['official_rows'] for s in native['samples'])} for m in ('audio','vision')},
     'native_full_sample_gate':{},
     'aligned_lineage':{m:dict(Counter(x['source_row_status'] for x in rows if x['modality']==m)) for m in ('audio','vision')},
     'sample02_navigation_candidates':sum(x['local_navigation_candidate'] for x in rows if x['sample_id']=='02'),
     'source_candidate_availability':availability['counts'],
     'media_origin_verified_samples':['02'],
     'formal_xai_mapping_status':'index_only','review_gate':'REVIEW_GATE_NO_FORMAL_SCOPE_CHANGE'}
for modality in ('audio','vision'):
    passed=[]
    failed=[]
    for sample in native['samples']:
        v=sample[modality]
        time_ok=all(r['native_time'][1]>r['native_time'][0] for r in v['strict_rows'])
        order_ok=(v['official_rows']<=1 or v['source_rows_strictly_increasing'] is True)
        success=(v['strict_unique_matches']==v['official_rows'] and order_ok
                 and v['all_zero_maxabs'] and time_ok)
        (passed if success else failed).append(sample['sample_id'])
    out['native_full_sample_gate'][modality]={'pass_count':len(passed),'pass_sample_ids':passed,
                                              'fail_count':len(failed),'fail_sample_ids':failed}
(HERE/'results/t2_aggregate_gate.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
