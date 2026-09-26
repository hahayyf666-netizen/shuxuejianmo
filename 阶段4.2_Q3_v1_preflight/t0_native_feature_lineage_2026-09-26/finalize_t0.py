"""Aggregate strict T0 source-row evidence without inventing local clip times."""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'results/t0_frozen_contract.json').read_text(encoding='utf-8'))
negative = json.loads((HERE / 'results/t0_wrong_video_negative_control.json').read_text(encoding='utf-8'))
ffprobe = HERE.parents[1] / 'work/tools/ffmpeg-9.0.2-essentials_build/ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe'
if not ffprobe.is_file():
    ffprobe = HERE.parents[2] / 'tools/ffmpeg-9.0.2-essentials_build/ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe'
assert ffprobe.is_file()

summary = {'source_row_gate': 'PASS_FROZEN_FIVE', 'attachment4_local_time_gate': 'BLOCKED_PENDING_CLIP_ORIGIN_VALIDATION',
           'eligible_for_20_sample_expansion': False, 'text_mapping_unchanged': True,
           'audio_vision_mapping_status_for_formal_xai': 'index_only', 'samples': [], 'totals': {}}
for frozen in contract['sources']:
    sid = frozen['sample_id']
    result = json.loads((HERE / 'results' / f't0_native_row_match_{sid}.json').read_text(encoding='utf-8'))
    cmd = [str(ffprobe), '-v', 'error', '-show_entries', 'format=duration,start_time', '-of', 'json', frozen['files']['aligned_mp4']['path']]
    media = json.loads(subprocess.check_output(cmd, text=True))['format']
    duration = float(media['duration'])
    record = {'sample_id': sid, 'candidate_video_id': result['candidate_video_id'],
              'clip_presentation_duration_sec': duration, 'clip_presentation_start_sec': float(media['start_time']),
              'audio': {}, 'vision': {}}
    for modality in ('audio', 'vision'):
        v = result['modalities'][modality]
        rows = v['strict_rows']
        count = v['official_shape'][0]
        order = all(rows[i]['native_r'] < rows[i+1]['native_r'] for i in range(len(rows)-1))
        time = all(r['native_time'][0] < r['native_time'][1] for r in rows)
        first, last = rows[0]['native_time'][0], rows[-1]['native_time'][1]
        span = last - first
        rec = {'matched': len(rows), 'official_rows': count, 'all_rows_strict_unique': len(rows) == count,
               'strict_source_order': order, 'positive_intervals': time,
               'source_time_first_sec': first, 'source_time_last_sec': last,
               'source_time_span_sec': span, 'span_within_clip_duration': span <= duration + 0.05,
               'clip_duration_minus_span_sec': duration - span,
               'all_values_exact_zero_maxabs': v['nearest_maxabs_min_median_max'][2] == 0.0,
               'source_row_gate': 'PASS' if len(rows)==count and order and time and span<=duration+0.05 and v['nearest_maxabs_min_median_max'][2]==0.0 else 'FAIL'}
        record[modality] = rec
        total = summary['totals'].setdefault(modality, {'matched': 0, 'official_rows': 0})
        total['matched'] += len(rows); total['official_rows'] += count
        if rec['source_row_gate'] != 'PASS':
            summary['source_row_gate'] = 'FAIL'
    summary['samples'].append(record)
for modality in ('audio', 'vision'):
    if negative['modalities'][modality]['strict_matches'] != 0:
        summary['source_row_gate'] = 'FAIL_NEGATIVE_CONTROL'
    summary['totals'][modality]['wrong_video_matches'] = negative['modalities'][modality]['strict_matches']
    summary['totals'][modality]['source_video_row_match_status'] = 'PASS' if summary['source_row_gate']=='PASS_FROZEN_FIVE' else 'FAIL'
(HERE / 'results/t0_aggregate_gate.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'source_row_gate': summary['source_row_gate'], 'local_time_gate': summary['attachment4_local_time_gate'], 'totals': summary['totals']}, indent=2))
