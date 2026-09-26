"""Conservatively connect sample02 C2/T0 rows to T1 media navigation times."""
import csv
import json
from collections import Counter
from pathlib import Path

import av

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
T0=json.loads((HERE.parent/'t0_native_feature_lineage_2026-09-26/results/t0_native_row_match_02.json').read_text())
T1=json.loads((HERE.parent/'t1_clip_origin_2026-09-26/results/t1_clip_origin_02.json').read_text())
FROZEN=json.loads((HERE.parent/'t1_clip_origin_2026-09-26/results/t1_frozen_contract.json').read_text())
clip_path=Path(FROZEN['source_media_for_02']['attachment4_clip']['path'])
c2_path=ROOT/'阶段4.2_Q3_v1_preflight/stage_c2_provenance/results/aligned_to_unaligned_row_matches_20.csv'
with c2_path.open(encoding='utf-8-sig',newline='') as handle:
    c2=[r for r in csv.DictReader(handle) if r['sample_id']=='02']
with av.open(str(clip_path)) as media:
    stream=media.streams.video[0]
    frame_pts=sorted(float(f.pts*stream.time_base) for f in media.decode(stream) if f.pts is not None)
audio_duration=T1['clip_audio_duration_sec']
video_offset=16.9
audio_offset=T1['full_peak_source_offset_sec']
rows=[]
for r in c2:
    modality=r['modality']
    seq=int(r['official_seq_index'])
    indices=[int(x) for x in r['exact_unaligned_indices_zero_based'].split(';') if x]
    record={'sample_id':'02','modality':modality,'official_seq_index':seq,
            'c2_exact_source_index_count':int(r['exact_source_index_count']),
            'unaligned_j':indices[0] if len(indices)==1 else None,
            'mapping_status':'index_only','reason':'C2_NOT_UNIQUE',
            'source_interval_start_sec':'','source_interval_end_sec':'',
            'local_interval_start_sec':'','local_interval_end_sec':'',
            'representative_local_video_pts_sec':'','nearby_frame_count':''}
    if len(indices)!=1 or int(r['exact_source_index_count'])!=1:
        rows.append(record); continue
    source_rows=T0['modalities'][modality]['strict_rows']
    j=indices[0]
    if j>=len(source_rows) or source_rows[j]['official_j']!=j:
        record['reason']='T0_SOURCE_ROW_MISSING'; rows.append(record); continue
    start,end=source_rows[j]['native_time']
    record['source_interval_start_sec']=f'{start:.6f}'
    record['source_interval_end_sec']=f'{end:.6f}'
    offset=audio_offset if modality=='audio' else video_offset
    local_start,local_end=start-offset,end-offset
    record['local_interval_start_sec']=f'{local_start:.6f}'
    record['local_interval_end_sec']=f'{local_end:.6f}'
    if modality=='audio':
        if local_start<0 or local_end>audio_duration or local_end<=local_start:
            record['reason']='AUDIO_CLIP_BOUNDARY'
        else:
            record['mapping_status']='reconstructed_media_navigation_candidate_only'
            record['reason']='C2_T0_T1_AUDIO_CHAIN_VALID'
    else:
        if local_start<0 or local_end>frame_pts[-1]+1/30 or local_end<=local_start:
            record['reason']='VIDEO_CLIP_BOUNDARY'
        else:
            mid=(local_start+local_end)/2
            close=[p for p in frame_pts if abs(p-mid)<=0.05]
            record['nearby_frame_count']=len(close)
            if not close:
                record['reason']='NO_DECODED_PTS_NEAR_INTERVAL'
            else:
                nearest=min(close,key=lambda p:abs(p-mid))
                record['representative_local_video_pts_sec']=f'{nearest:.6f}'
                record['mapping_status']='reconstructed_media_navigation_candidate_only'
                record['reason']='C2_T0_T1_VIDEO_NAVIGATION_ONLY'
    rows.append(record)
out=HERE/'results/sample02_row_navigation.csv'
with out.open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
summary={'sample_id':'02','total_c2_rows':len(rows),'clip_audio_duration_sec':audio_duration,
         'video_decoded_pts_count':len(frame_pts),'video_first_last_pts_sec':[frame_pts[0],frame_pts[-1]],
         'by_modality':{m:dict(Counter(r['reason'] for r in rows if r['modality']==m)) for m in ('audio','vision')},
         'formal_xai_mapping_status':'index_only'}
(HERE/'results/sample02_row_navigation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
