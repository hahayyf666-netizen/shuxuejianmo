from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys, time, zipfile
from collections import Counter
from pathlib import Path
import av
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def decoded_times(p):
    vp=[]; vd=[]
    with av.open(str(p)) as c:
        stream=c.streams.video[0]
        for frame in c.decode(stream):
            if frame.pts is None:raise ValueError('source video missing PTS')
            vp.append(float(frame.pts*frame.time_base))
            vd.append(float(frame.duration*frame.time_base) if frame.duration else None)
    video_end=vp[-1]+(vd[-1] or (vp[-1]-vp[-2] if len(vp)>1 else 0))
    output=[]; raw_next=None; raw_gap=0.; pcm_peak=0.; pcm_sum_squares=0.; pcm_samples=0
    resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
    def collect(frames):
        nonlocal pcm_peak,pcm_samples,pcm_sum_squares
        for out in (frames or []):
            output.append((float(out.pts*out.time_base),out.samples))
            values=out.to_ndarray().astype(np.float64,copy=False)
            pcm_peak=max(pcm_peak,float(np.max(np.abs(values))))
            pcm_sum_squares+=float(np.square(values).sum());pcm_samples+=values.size
    with av.open(str(p)) as c:
        for frame in c.decode(c.streams.audio[0]):
            pt=float(frame.pts*frame.time_base)
            if raw_next is not None:raw_gap=max(raw_gap,abs(pt-raw_next))
            raw_next=pt+frame.samples/frame.sample_rate
            collect(resampler.resample(frame))
        collect(resampler.resample(None))
    residual=max([abs(output[i][0]-(output[i-1][0]+output[i-1][1]/16000)) for i in range(1,len(output))] or [0.])
    return np.asarray(vp),video_end,np.asarray([output[0][0],output[-1][0]+output[-1][1]/16000]),raw_gap,residual,pcm_peak,pcm_sum_squares/max(pcm_samples,1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--report-root',type=Path,required=True)
    args=parser.parse_args();project=args.project_root.resolve();out=args.report_root.resolve();out.mkdir(parents=True,exist_ok=True)
    root=project/'outputs/q1/v1_delivery';audit=project/'outputs/q1/diagnostics/full100_correspondence_audit'
    env=read_json(root/'environment.json');input_root=Path(env['input_root_argument'])
    spec=importlib.util.spec_from_file_location('q1_baseline_validator',root/'code/validate_q1_v1.py');validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    labels,label_sha=validator.read_official_labels(input_root/'label-100.xlsx')
    manifest=rows(root/'manifest.csv');inputs={r['sample_key']:r for r in rows(root/'input_manifest.csv')};audits={r['sample_key']:r for r in rows(audit/'audit_100.csv')};assets=read_json(root/'metadata/model_assets.json')
    protected=list((root/'features').glob('*.npz'))+list((root/'code').glob('*.py'))+list(root.glob('*.md'))+list(root.glob('*.csv'))+list(root.glob('*.json'))+list((root/'reports').glob('*.*'))
    before={str(p.relative_to(project)):digest(p) for p in protected if p.is_file()}
    result=[];failures=[];decoded_zero=[];started=time.perf_counter()
    for i,row in enumerate(manifest):
        key=row['sample_key'];p=root/row['output_file'];rec={'sample_key':key,'output_file':row['output_file']};checks={}
        try:
            validator.validate_one(root,input_root,row,inputs[key],labels,label_sha,audits,audit,assets)
            checks['existing_validate_one']=True
        except Exception as exc:checks['existing_validate_one']=False;rec['existing_validator_error']=str(exc)
        try:
            with np.load(p,allow_pickle=False) as z:
                checks['official_text_matches_workbook']=str(z['official_text'].item())==labels[key]
                checks['all_manifest_flags_match_npz']=all(int(row[name])==int(z[name].item()) for name in ['text_present','text_content_valid','audio_present','audio_observation_valid','audio_speech_valid','video_present','audio_visual_time_valid','vision_feature_valid','face_feature_valid','text_sequence_length','audio_observation_length','video_observation_length'])
                audit_state=audits[key]['text_audio_correspondence']
                canonical_state={'confirmed_match':'confirmed_match','confirmed_mismatch':'confirmed_mismatch','no_speech':'no_speech','unknown':'not_asserted'}
                checks['correspondence_matches_manifest_and_audit']=audit_state in canonical_state and str(z['text_audio_correspondence'].item())==row['text_audio_correspondence']==canonical_state[audit_state]
                vpts,vend,acov,raw_gap,pcm_gap,peak,energy=decoded_times(input_root/row['source_relpath'])
                checks['all_source_video_pts_match']=len(vpts)==len(z['raw_video_pts_sec']) and np.allclose(vpts,z['raw_video_pts_sec'],atol=1e-10,rtol=0)
                checks['source_video_coverage_matches']=np.allclose([vpts[0],vend],z['video_presentation_coverage_sec'],atol=1e-10,rtol=0)
                checks['source_audio_coverage_matches']=np.allclose(acov,z['audio_presentation_coverage_sec'],atol=1e-10,rtol=0)
                checks['source_audio_pts_contiguous']=raw_gap<1e-6 and pcm_gap<1e-8
                checks['declared_silence_has_zero_pcm']=int(z['audio_speech_valid'].item())!=0 or peak==0
                checks['audio_windows_in_decoded_coverage']=bool(np.all(z['raw_audio_lld_start_sec']>=acov[0]-1e-6) and np.all(z['raw_audio_lld_end_sec']<=acov[1]+1e-6))
                expected_ws=np.maximum(vpts[0],np.r_[vpts[0],(vpts[:-1]+vpts[1:])/2])
                expected_we=np.r_[(vpts[:-1]+vpts[1:])/2,vend]
                checks['video_support_convention_recomputes']=np.allclose(expected_ws,z['raw_video_support_start_sec'],atol=1e-10,rtol=0) and np.allclose(expected_we,z['raw_video_support_end_sec'],atol=1e-10,rtol=0)
                token_offsets=z['roberta_token_offsets'];checks['token_offsets_within_official_text']=bool(np.all(token_offsets>=0) and np.all(token_offsets<=len(labels[key])))
                checks['invalid_word_times_all_nan']=bool(np.isnan(z['word_start_sec'][z['word_time_valid']==0]).all() and np.isnan(z['word_end_sec'][z['word_time_valid']==0]).all())
                checks['placeholder_masks_preserved']=bool(np.all(z['word_audio_feat'][z['word_audio_valid']==0]==0) and np.all(z['word_vision_feat'][z['word_vision_valid']==0]==0) and np.all(z['raw_video_blendshape_values'][z['raw_video_face_feature_valid']==0]==0))
                stub=p.stem;ml=read_json(root/'logs'/f'{stub}.media.json');tl=read_json(root/'logs'/f'{stub}.text.json')
                checks['per_sample_logs_match_identity']=ml['sample_key']==tl['sample_key']==key and ml['status']==tl['status']=='PASS' and ml['output_sha256']==row['output_sha256']
                rec.update(alignment_mode=row['alignment_mode'],text_audio_correspondence=row['text_audio_correspondence'],audio_speech_valid=int(z['audio_speech_valid'].item()),words=len(z['words']),audio_windows=len(z['raw_audio_lld_values']),video_frames=len(vpts),face_frames=int(z['raw_video_face_feature_valid'].sum()),valid_word_times=int(z['word_time_valid'].sum()),audio_word_count=int(z['word_audio_valid'].sum()),vision_word_count=int(z['word_vision_valid'].sum()),roberta_tokens=len(z['roberta_token_ids']),pcm_peak=peak,pcm_mean_square=energy,raw_audio_max_pts_gap=raw_gap,resampled_max_pts_gap=pcm_gap,lld_first_width_sec=float(z['raw_audio_lld_end_sec'][0]-z['raw_audio_lld_start_sec'][0]),lld_last_width_sec=float(z['raw_audio_lld_end_sec'][-1]-z['raw_audio_lld_start_sec'][-1]),no_padding=(len(z['words'])==int(z['text_sequence_length'].item()) and len(vpts)==int(z['video_observation_length'].item())))
                if peak==0:decoded_zero.append(key)
        except Exception as exc:checks['additional_audit_completed']=False;rec['additional_audit_error']=f'{type(exc).__name__}: {exc}'
        rec['checks']=checks;rec['all_checks_pass']=all(checks.values());result.append(rec)
        if not rec['all_checks_pass']:failures.append({'sample_key':key,'failed_checks':[n for n,b in checks.items() if not b],'error':rec.get('additional_audit_error',rec.get('existing_validator_error',''))})
        if (i+1)%10==0:print(f'Audited {i+1}/100; sample check failures={len(failures)}',flush=True)
    package=read_json(root/'package/package_manifest.json');zip_path=root/'package/q1_v1_candidate.zip'
    with zipfile.ZipFile(zip_path) as zf:
        package_checks={'archive_sha256_matches':digest(zip_path)==package['archive_sha256'],'crc_pass':zf.testzip() is None,'member_set_matches':set(zf.namelist())=={f['path'] for f in package['files']},'all_member_hashes_match':all(hashlib.sha256(zf.read(f['path'])).hexdigest()==f['sha256'] for f in package['files'])}
    changed=[str(p.relative_to(project)) for p in protected if p.is_file() and before.get(str(p.relative_to(project)))!=digest(p)]
    summary={'scope':'Read-only source-to-artifact requirement re-audit; no new feature extraction or alignment','official_label_sha256':label_sha,'input_root':str(input_root),'python':sys.version,'av_version':av.__version__,'official_label_count':len(labels),'manifest_count':len(manifest),'npz_count':len(list((root/'features').glob('*.npz'))),'all_sample_keys_match':set(labels)==set(inputs)=={r['sample_key'] for r in manifest},'manifest_results_bytes_identical':(root/'manifest.csv').read_bytes()==(root/'results_100.csv').read_bytes(),'sample_checks_pass_count':sum(r['all_checks_pass'] for r in result),'failures':failures,'counts':{n:sum(r.get(n,0) for r in result) for n in ['words','audio_windows','video_frames','face_frames','valid_word_times','audio_word_count','vision_word_count']},'modes':dict(Counter(r['alignment_mode'] for r in result)),'no_face_samples':sum(r.get('face_frames',0)==0 for r in result),'max_roberta_tokens':max(r.get('roberta_tokens',0) for r in result),'decoded_digital_silence_samples':decoded_zero,'package_checks':package_checks,'package_members':package['candidate_file_count'],'archive_sha256':package['archive_sha256'],'protected_file_count':len(before),'protected_files_changed':changed,'elapsed_seconds':time.perf_counter()-started,'samples':result}
    (out/'machine_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'input_snapshot_sha256.json').write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    keys=['sample_key','output_file','alignment_mode','text_audio_correspondence','audio_speech_valid','words','audio_windows','video_frames','face_frames','valid_word_times','audio_word_count','vision_word_count','roberta_tokens','pcm_peak','all_checks_pass']
    with (out/'sample_checks_100.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(result)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['samples','input_root','python']},ensure_ascii=False),flush=True)
    return int(bool(failures or changed or not all(package_checks.values())))

if __name__=='__main__':raise SystemExit(main())
