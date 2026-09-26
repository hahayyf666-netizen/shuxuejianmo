"""Freeze T1 media inputs and decision rules before waveform similarity values."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ASSETS = WORK / 'q3_native_t0_assets'
T0 = HERE.parent / 't0_native_feature_lineage_2026-09-26/results'
frozen = json.loads((T0/'t0_frozen_contract.json').read_text(encoding='utf-8'))
sample = next(x for x in frozen['sources'] if x['sample_id']=='02')
files = {'attachment4_clip': Path(sample['files']['aligned_mp4']['path']),
         'source_audio': ASSETS/'YCEllKyaCrc_source_audio.webm',
         'source_video': ASSETS/'YCEllKyaCrc_source_video.mp4'}
for p in files.values():
    if not p.is_file(): raise FileNotFoundError(p)
def meta(p): return {'path':str(p), 'bytes':p.stat().st_size,
                    'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
doc = {'status':'FROZEN_BEFORE_WAVEFORM_VALUES', 'sample_ids':['02','03'],
       '02_source_video_id':'YCEllKyaCrc',
       '03_source_video_id':'GAVpYuhMZAw',
       '03_source_media_status':'YouTube metadata unavailable (oEmbed 404, yt-dlp video unavailable); do not infer offset',
       'source_media_for_02':{k:meta(v) for k,v in files.items()},
       'source_download':{'url':'https://www.youtube.com/watch?v=YCEllKyaCrc',
                          'yt_dlp_version':'2026.08.19','audio_format_id':'251-7',
                          'video_format_id':'134',
                          'oembed_title':'Charlie Chaplin - Final Speech from The Great Dictator (Clip)'},
       'audio_rule':{'decode':'ffmpeg first audio stream to mono 16000-Hz float32 PCM; use entire clip; no text/feature time in search',
                     'search':'normalized Pearson sliding correlation over whole 63-second source',
                     'peak_at_least':0.6,'peak_minus_best_outside_one_second_at_least':0.10,
                     'half_clip_peak_at_least':0.5,
                     'half_clip_offsets_agree_with_full_within_sec':0.12,
                     'source_negative_outside_one_second_below':0.5},
       'vision_rule':{'clip_sample_times_sec':[0.5,1.5,2.5],
                       'decode':'ffmpeg selected frames to 64x64 gray using presentation timestamps',
                       'compare':'Pearson pixel correlation at audio-derived source offset + clip sample time; allow nearest source frame within 0.08 s only',
                       'at_least_two_frame_correlations_above':0.55,
                       'do_not_fit_offset_to_visual_or_CSD_feature_times':True},
       'gate_rule':'PASS_LOCAL only if source identity and audio and vision rules pass; otherwise BLOCKED; source CSD time converted only after PASS_LOCAL'}
(HERE/'results').mkdir(parents=True,exist_ok=True)
(HERE/'results/t1_frozen_contract.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print('frozen', {k:v['sha256'] for k,v in doc['source_media_for_02'].items()})
