from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, re, sys, zipfile
from collections import Counter
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def lines(p,needle):
    return [{'line':i,'text':s} for i,s in enumerate(Path(p).read_text(encoding='utf-8').splitlines(),1) if needle in s]

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--project-root',required=True,type=Path);pa.add_argument('--report-root',required=True,type=Path);args=pa.parse_args()
    project=args.project_root.resolve();out=args.report_root.resolve();root=project/'outputs/q1/v1_delivery';assets=read(root/'metadata/model_assets.json');env=read(root/'environment.json')
    out.mkdir(parents=True,exist_ok=True)
    assetroot=project/'work/model_assets_a0';model=assetroot/'roberta-base'/assets['roberta_revision']
    model_checks={name:sha(model/name)==expected for name,expected in assets['roberta_files_sha256'].items()}
    model_checks['mediapipe']=sha(assetroot/assets['mediapipe_model_file'])==assets['mediapipe_model_sha256']
    a0=read(root/'metadata/protected_a0_baseline.json');a0_fail=[p for p,s in a0.items() if not (project/p).is_file() or sha(project/p)!=s]
    packages={name:{'recorded':env['package_'+name.replace('-','_')],'installed':importlib.metadata.version(name)} for name in ['numpy','torch','stable-ts','openai-whisper','opensmile','mediapipe','transformers','av','pandas','openpyxl']}
    package_versions_match=all(x['recorded']==x['installed'] for x in packages.values())
    with zipfile.ZipFile(root/'package/q1_v1_candidate.zip') as z:
        names=set(z.namelist())
        req_path='outputs/q1/v1_delivery/requirements-lock.txt';req=z.read(req_path).decode('utf-8-sig').splitlines()
        local_refs=[{'line':i,'requirement':s,'distribution_in_zip':any(Path(n).name==s.split('/')[-1].split('#')[0] for n in names)} for i,s in enumerate(req,1) if '@ file:' in s]
        upstream=[{'path':p,'in_zip':p in names,'exists_in_workspace':(project/p).exists(),'sha256':sha(project/p)} for p in ['work/run_a0_pilot_worker.py','work/run_prefreeze_audio_alignment.py','work/run_additional_controls.py','work/run_full100_alignment_audit.py']]
        align_refs=[]
        for name in sorted(names):
            if re.fullmatch(r'outputs/q1/diagnostics/full100_correspondence_audit/alignment/[^/]+\.json',name):
                data=json.loads(z.read(name));target=data.get('trace_relative_path','').replace('\\','/')
                align_refs.append({'record':name,'trace':target,'exists':target in names,'sha_matches':target in names and hashlib.sha256(z.read(target)).hexdigest()==data.get('trace_sha256')})
        workbook_assets_excluded=not any(n.endswith('.mp4') or n.endswith('label-100.xlsx') or n.endswith('model.safetensors') for n in names)
        required_logs=['outputs/q1/v1_delivery/logs/'+p.stem+s for p in (root/'features').glob('*.npz') for s in ['.media.json','.text.json']]
        logs_present=sum(n in names for n in required_logs)
    site=project/'work/.venv_q1/Lib/site-packages'
    rootconf=site/'opensmile/core/config/egemaps/v02/eGeMAPSv02.conf'
    conf=site/'opensmile/core/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc'
    egemaps=site/'opensmile/core/config/egemaps/v02/eGeMAPSv02_core.lld.conf.inc'
    smile=site/'opensmile/core/smile.py'
    mp=site/'mediapipe/tasks/python/vision/face_landmarker.py'
    evidence={str(p.relative_to(project)):{'sha256':sha(p),'snippets':sum([lines(p,needle) for needle in needles],[])} for p,needles in [(rootconf,['core.lld.conf.inc']),(conf,['frameSize =','frameStep =','frameCenterSpecial =']),(egemaps,['smaWin =']),(smile,['starts.append(meta.time)','ends.append(meta.time + meta.lengthSec)']),(mp,['min_face_detection_confidence:','min_face_presence_confidence:','min_tracking_confidence:'])]}
    result={'scope':'Read-only environment/package/upstream implementation audit; no dependency installation or inference','model_hash_checks':model_checks,'a0_protected_count':len(a0),'a0_hash_failures':a0_fail,'package_versions':packages,'package_versions_match':package_versions_match,'openSMILE_root_config_hash_matches':sha(rootconf)==assets['opensmile_config_sha256'],'pip_absolute_file_references':local_refs,'upstream_alignment_generators':upstream,'alignment_summary_count':len(align_refs),'all_alignment_trace_files_present_and_hash_match':all(x['exists'] and x['sha_matches'] for x in align_refs),'alignment_traces':align_refs,'per_sample_text_and_media_logs_in_zip':logs_present,'expected_sample_log_count':len(required_logs),'raw_media_and_large_model_assets_excluded_as_declared':workbook_assets_excluded,'time_coordinate_semantics':evidence,'source_only_command':'python work/q1_repro_contract_audit.py --project-root . --report-root outputs/q1/v1_delivery/reports/requirement_alignment_check_2026-09-24','fresh_environment_install_run':False,'failing_requirement':'Portable complete reproduction instructions','overall_requirement_status':'NEEDS_REVISION'}
    (out/'reproduction_and_semantics_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['model_hash_checks','a0_protected_count','a0_hash_failures','package_versions_match','openSMILE_root_config_hash_matches','per_sample_text_and_media_logs_in_zip','alignment_summary_count','all_alignment_trace_files_present_and_hash_match','overall_requirement_status']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
