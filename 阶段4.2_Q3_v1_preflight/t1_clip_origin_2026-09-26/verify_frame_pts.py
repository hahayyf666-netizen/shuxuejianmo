"""Validate source/clip visual frame correspondence using decoded presentation PTS."""
import json
from pathlib import Path

import av
import numpy as np

HERE=Path(__file__).resolve().parent
frozen=json.loads((HERE/'results/t1_frozen_contract.json').read_text(encoding='utf-8'))
result=json.loads((HERE/'results/t1_clip_origin_02.json').read_text(encoding='utf-8'))
paths={k:Path(v['path']) for k,v in frozen['source_media_for_02'].items()}

def frames(path):
    out=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.time_base
        for frame in container.decode(stream):
            if frame.pts is None: continue
            pts=float(frame.pts*stream.time_base)
            array=frame.reformat(width=64,height=64,format='gray').to_ndarray().astype(np.float64).reshape(-1)
            array-=array.mean()
            norm=np.linalg.norm(array)
            if norm>0: out.append((pts,array/norm))
    return out

src=frames(paths['source_video'])
clip=frames(paths['attachment4_clip'])
audio_offset=result['full_peak_source_offset_sec']
checks=[]
for requested in frozen['vision_rule']['clip_sample_times_sec']:
    cp,cvec=min(clip,key=lambda x:abs(x[0]-requested))
    candidates=[(sp,float(np.dot(cvec,svec))) for sp,svec in src if abs(sp-(audio_offset+cp))<=0.08]
    if not candidates: raise RuntimeError('no source frame near audio offset')
    sp,score=max(candidates,key=lambda x:x[1])
    checks.append({'requested_clip_sec':requested,'actual_clip_pts_sec':cp,'matched_source_pts_sec':sp,
                   'presentation_pts_offset_sec':sp-cp,'pixel_pearson':score,
                   'candidate_count':len(candidates)})
offsets=np.array([x['presentation_pts_offset_sec'] for x in checks])
summary={'sample_id':'02','source_frame_count':len(src),'clip_frame_count':len(clip),
         'audio_offset_sec':audio_offset,'frames':checks,
         'video_offset_median_sec':float(np.median(offsets)),
         'video_offset_range_sec':float(np.ptp(offsets)),
         'video_minus_audio_offset_sec':float(np.median(offsets)-audio_offset),
         'all_visual_correlations_above_frozen_threshold':all(x['pixel_pearson']>=0.55 for x in checks),
         'video_offset_stable_within_one_source_frame':bool(np.ptp(offsets)<=0.04)}
all_matches=[]
for cp,cvec in clip:
    candidates=[(sp,float(np.dot(cvec,svec))) for sp,svec in src if abs(sp-(audio_offset+cp))<=0.08]
    if not candidates: continue
    sp,score=max(candidates,key=lambda x:x[1])
    all_matches.append({'clip_pts_sec':cp,'source_pts_sec':sp,'offset_sec':sp-cp,'pixel_pearson':score})
all_offsets=np.array([x['offset_sec'] for x in all_matches])
all_scores=np.array([x['pixel_pearson'] for x in all_matches])
summary['all_decoded_frames_diagnostic']={'matched':len(all_matches),'total':len(clip),
    'offset_median_sec':float(np.median(all_offsets)),
    'offset_min_sec':float(np.min(all_offsets)), 'offset_max_sec':float(np.max(all_offsets)),
    'pearson_min':float(np.min(all_scores)), 'pearson_median':float(np.median(all_scores)),
    'pearson_below_0_55':int(np.sum(all_scores<0.55))}
(HERE/'results/t1_frame_pts_02.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
