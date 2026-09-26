"""Freeze top-candidate native row gate before expanding beyond T0's five."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
candidates=json.loads((HERE/'results/source_candidates_20.json').read_text(encoding='utf-8'))
assert len(candidates['samples'])==20
doc={'status':'FROZEN_BEFORE_REMAINING_NATIVE_CANDIDATE_VALUES',
     'candidate_rule':'Only the top transcript CSD video group per sample; no model outputs or labels; no per-sample offset tuning',
     'source_mirror_commit':'ee52115996266573d2e3c2a3e04fd19e52af74c8',
     'sample_to_candidate':{x['sample_id']:(x['top_candidates'][0]['video_id'] if x['top_candidates'] else None) for x in candidates['samples']},
     'numeric_rule':{'same_dimension':True,'maxabs_at_most':1e-4,'relative_l2_at_most':1e-6,
                     'second_best_maxabs_at_least':1e-3,'all_valid_rows_required':True,
                     'strictly_increasing_source_rows':True,'positive_source_intervals':True},
     'claim_boundary':'An exact CSD source row is not an Attachment4 local timestamp; media-origin and C2 chain remain separate gates',
     'no_model_or_prediction_run':True}
(HERE/'results/native20_frozen_rules.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print('frozen',len(doc['sample_to_candidate']))
