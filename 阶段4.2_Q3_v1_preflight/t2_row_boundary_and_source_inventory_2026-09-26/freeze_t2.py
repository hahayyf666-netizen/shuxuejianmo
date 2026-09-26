"""Freeze bounded row-level navigation check and source availability inventory."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
T0=HERE.parent/'t0_native_feature_lineage_2026-09-26/results'
T1=HERE.parent/'t1_clip_origin_2026-09-26/results'
assert json.loads((T1/'t1_aggregate_gate.json').read_text())['status']=='PARTIAL_PASS_MEDIA_ORIGIN'
doc={'status':'FROZEN_BEFORE_ROW_OUTPUT', 'sample_02_rule':{
    'audio':'source CSD interval minus independently measured 16.8366875s audio offset; require positive interval fully inside decoded clip audio duration',
    'vision':'source CSD interval minus independently measured 16.9s median of three preselected video PTS anchors; negative-start or out-of-clip interval is boundary/unavailable; choose nearest actually decoded local PTS to midpoint within 0.05s; report candidate count within 0.05s and never assert official extractor frame',
    'c2':'official aligned content index must have exactly one C-2 exact source index; otherwise unavailable',
    'mapping_label':'reconstructed_media_navigation_candidate_only',
    'formal_mapping_status':'index_only'},
    'availability_inventory_rule':{'scope':'20 Attachment4 unaligned raw_text samples',
      'candidate_id':'TimestampedWords 3-gram lookup only; candidate, never provenance',
      'availability':'YouTube oEmbed HTTP status only; no full media download or XAI prediction',
      'candidate_minimum':'report trigram hits and exact ordered token hits; zero/weak IDs remain unverified',
      'source_lineage':'no candidate ID upgraded without strict native CSD row match'},
    'no_model_or_prediction_run':True}
(HERE/'results').mkdir(parents=True,exist_ok=True)
(HERE/'results/t2_frozen_rules.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print('frozen')
