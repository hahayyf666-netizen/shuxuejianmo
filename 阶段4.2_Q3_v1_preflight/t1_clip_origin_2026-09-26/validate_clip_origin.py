"""Media-first T1 validation for frozen Attachment4 sample 02."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
frozen = json.loads((HERE/'results/t1_frozen_contract.json').read_text(encoding='utf-8'))
ffmpeg = HERE.parents[2] / 'tools/ffmpeg-9.0.2-essentials_build/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe'
assert ffmpeg.is_file()
paths = {k:Path(v['path']) for k,v in frozen['source_media_for_02'].items()}
for k,p in paths.items():
    assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest() == frozen['source_media_for_02'][k]['sha256']

def pcm(path):
    cmd=[str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
    raw=subprocess.check_output(cmd)
    return np.frombuffer(raw,dtype='<f4').astype(np.float64)

source=pcm(paths['source_audio'])
clip=pcm(paths['attachment4_clip'])
sr=16000
def scan(x,y):
    y=y-y.mean()
    n=len(y)
    numerator=fftconvolve(x,y[::-1],mode='valid')
    cs=np.r_[0.0,np.cumsum(x)]
    cs2=np.r_[0.0,np.cumsum(x*x)]
    energy=(cs2[n:]-cs2[:-n])-(cs[n:]-cs[:-n])**2/n
    denom=np.sqrt(np.maximum(energy,1e-12))*np.linalg.norm(y)
    corr=numerator/denom
    arg=int(np.argmax(corr))
    return corr,arg,float(corr[arg])

corr,idx,peak=scan(source,clip)
away=np.ones(len(corr),dtype=bool)
away[max(0,idx-sr):min(len(corr),idx+sr+1)]=False
second=float(np.max(corr[away]))
half=len(clip)//2
first_c, first_i, first_peak=scan(source,clip[:half])
second_c, second_i, second_peak=scan(source,clip[half:])
second_offset=second_i-half
audio_rule=frozen['audio_rule']
audio_pass=bool(peak>=audio_rule['peak_at_least'] and peak-second>=audio_rule['peak_minus_best_outside_one_second_at_least']
                and second<audio_rule['source_negative_outside_one_second_below']
                and first_peak>=audio_rule['half_clip_peak_at_least'] and second_peak>=audio_rule['half_clip_peak_at_least']
                and abs(first_i-idx)/sr<=audio_rule['half_clip_offsets_agree_with_full_within_sec']
                and abs(second_offset-idx)/sr<=audio_rule['half_clip_offsets_agree_with_full_within_sec'])
out={'sample_id':'02','source_id':'YCEllKyaCrc','sample_rate':sr,
     'source_audio_duration_sec':len(source)/sr,'clip_audio_duration_sec':len(clip)/sr,
     'full_peak_corr':peak,'best_other_corr_outside_one_sec':second,'peak_margin':peak-second,
     'full_peak_source_offset_sec':idx/sr,
     'first_half_peak_corr':first_peak,'first_half_offset_sec':first_i/sr,
     'second_half_peak_corr':second_peak,'second_half_offset_sec':second_offset/sr,
     'audio_rule_pass':audio_pass,'vision':[]}
print('audio', {k:out[k] for k in ('full_peak_corr','best_other_corr_outside_one_sec','peak_margin','full_peak_source_offset_sec','first_half_peak_corr','first_half_offset_sec','second_half_peak_corr','second_half_offset_sec','audio_rule_pass')},flush=True)

def gray_frame(path,t):
    cmd=[str(ffmpeg),'-hide_banner','-loglevel','error','-ss',f'{t:.6f}','-i',str(path),'-frames:v','1',
         '-vf','scale=64:64,format=gray','-f','rawvideo','pipe:1']
    raw=subprocess.check_output(cmd)
    return np.frombuffer(raw,dtype=np.uint8).astype(np.float64) if len(raw)==4096 else None

if audio_pass:
    for t in frozen['vision_rule']['clip_sample_times_sec']:
        a=gray_frame(paths['attachment4_clip'],t)
        candidates=[]
        for delta in (-0.08,-0.04,0.0,0.04,0.08):
            b=gray_frame(paths['source_video'],idx/sr+t+delta)
            if a is not None and b is not None and np.std(a)>1 and np.std(b)>1:
                candidates.append({'source_time_sec':idx/sr+t+delta,
                                   'pixel_pearson':float(np.corrcoef(a,b)[0,1])})
        best=max(candidates,key=lambda z:z['pixel_pearson']) if candidates else None
        out['vision'].append({'clip_time_sec':t,'best':best,'candidates':candidates})
    scores=[v['best']['pixel_pearson'] for v in out['vision'] if v['best']]
    out['vision_rule_pass']=sum(x>=frozen['vision_rule']['at_least_two_frame_correlations_above'] for x in scores)>=2
else:
    out['vision_rule_pass']=False
out['local_time_gate']='PASS_SAMPLE_02' if audio_pass and out['vision_rule_pass'] else 'BLOCKED'
(HERE/'results/t1_clip_origin_02.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('vision',[(x['clip_time_sec'], x['best']) for x in out['vision']],flush=True)
print('gate',out['local_time_gate'],flush=True)
