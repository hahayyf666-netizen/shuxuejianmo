# T4 Raw Evidence Closure Report

- Gate: **T4_COMPLETE_WITH_LIMITATIONS**
- Starting HEAD: `5454444328a08e3d5b59212478b33f20b694c221`
- Frozen scope: Attachment4 samples 04, 14, 19 only.
- Training / model selection / inference / test evaluation: **NO**.
- Shapley / IG / formal predictions changed: **NO**.
- Historical 4.6 Gate changed: **NO**; 4.7 entered: **NO**.
- Mapping interpretation: reconstructed raw-media navigation evidence; it does not prove the exact frame/window originally sampled by the official extractor.
- T4 selected-frame comparison reuses the frozen T3 numeric cutoff `min_two_correlations` = `0.55` on the individual source/local frame pair. This is a per-position check; it is not the T3 aggregate rule (at least 2 of 3 anchor comparisons pass, with offset range ≤ `0.12 s`).
- T4 frame preprocessing reuses T3: all presentation PTS via PyAV; grayscale 64×64; Pearson computed as the dot product of mean-centered unit vectors.

## Per-sample results

| Sample | Vision evidence units | Vision positions | Audio union positions | Audio mapped | Video offset (s) | Audio offset (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 04 | 3/3 | 15,14,13 | 2/4 | 25,26,1,15 | 179.03 | 178.9681875 | vision=PASS; audio=PARTIAL |
| 14 | 2/2 | 34,35,33 | 4/4 | 11,12,13,10 | 20.292666666666666 | 20.2025 | vision=PASS; audio=PASS |
| 19 | 3/3 | 22,21,23 | 5/5 | 22,7,25,21,12 | 170.4 | 170.347375 | vision=PASS; audio=PASS |

## Visual position details

| Sample | Position | Unaligned row | FACET interval (s) | Local interval (s) | Candidate frames | Selected PTS | Source PTS | Pearson | Same unit | Mapping | Failure |
|---|---:|---:|---|---|---:|---:|---:|---:|---|---|---|
| 04 | 15 | 96 | 185.4–185.433 | 6.3700000000000045–6.402999999999992 | 1 | 6.4 | 185.435 | 0.9994640707622817 | False | reconstructed_keyframe_from_verified_lineage |  |
| 04 | 14 | 95 | 185.333–185.367 | 6.302999999999997–6.336999999999989 | 1 | 6.333333333333333 | 185.352 | 0.9994841373882892 | False | reconstructed_keyframe_from_verified_lineage |  |
| 04 | 13 | 94 | 185.267–185.3 | 6.236999999999995–6.27000000000001 | 1 | 6.266666666666667 | 185.31 | 0.9994637778424708 | False | reconstructed_keyframe_from_verified_lineage |  |
| 14 | 34 | 152 | 30.3667–30.4 | 10.074033333333336–10.107333333333333 | 1 | 10.1 | 30.397 | 0.9994095812318713 | True | reconstructed_keyframe_from_verified_lineage |  |
| 14 | 35 | 152 | 30.3667–30.4 | 10.074033333333336–10.107333333333333 | 1 | 10.1 | 30.397 | 0.9994095812318713 | True | reconstructed_keyframe_from_verified_lineage |  |
| 14 | 33 | 151 | 30.3–30.3333 | 10.007333333333335–10.040633333333336 | 1 | 10.033333333333333 | 30.33 | 0.9993982517711786 | False | reconstructed_keyframe_from_verified_lineage |  |
| 19 | 22 | 189 | 183.0–183.033 | 12.599999999999994–12.632999999999981 | 1 | 12.6 | 183.0 | 0.9998167042682429 | False | reconstructed_keyframe_from_verified_lineage |  |
| 19 | 21 | 188 | 182.933–182.967 | 12.532999999999987–12.567000000000007 | 2 | 12.533333333333333 | 182.933 | 0.9998233731390763 | False | reconstructed_keyframe_from_verified_lineage |  |
| 19 | 23 | 190 | 183.067–183.1 | 12.667000000000002–12.699999999999989 | 1 | 12.7 | 183.1 | 0.9998168361180464 | False | reconstructed_keyframe_from_verified_lineage |  |

## Audio position details

| Sample | Target | Position | Unaligned row | COVAREP interval (s) | Audio offset (s) | Local speech interval (s) | Mapping | Failure |
|---|---|---:|---:|---|---:|---|---|---|
| 04 | classification | 25 | 136 | 185.81–185.82 | 178.9681875 | 6.841812500000003–6.851812499999994 | index_only | mapped local speech interval lies outside T3-recorded Attachment4 audio duration |
| 04 | classification | 26 | 136 | 185.81–185.82 | 178.9681875 | 6.841812500000003–6.851812499999994 | index_only | mapped local speech interval lies outside T3-recorded Attachment4 audio duration |
| 04 | classification | 1 | 117 | 184.86–184.87 | 178.9681875 | 5.8918125000000146–5.9018125000000055 | reconstructed_speech_interval_from_verified_lineage |  |
| 04 | regression | 15 | 130 | 185.51–185.51999999999998 | 178.9681875 | 6.541812499999992–6.551812499999983 | reconstructed_speech_interval_from_verified_lineage |  |
| 04 | regression | 25 | 136 | 185.81–185.82 | 178.9681875 | 6.841812500000003–6.851812499999994 | index_only | mapped local speech interval lies outside T3-recorded Attachment4 audio duration |
| 04 | regression | 26 | 136 | 185.81–185.82 | 178.9681875 | 6.841812500000003–6.851812499999994 | index_only | mapped local speech interval lies outside T3-recorded Attachment4 audio duration |
| 14 | classification | 11 | 187 | 29.58–29.59 | 20.2025 | 9.377499999999998–9.3875 | reconstructed_speech_interval_from_verified_lineage |  |
| 14 | classification | 12 | 187 | 29.58–29.59 | 20.2025 | 9.377499999999998–9.3875 | reconstructed_speech_interval_from_verified_lineage |  |
| 14 | classification | 13 | 187 | 29.58–29.59 | 20.2025 | 9.377499999999998–9.3875 | reconstructed_speech_interval_from_verified_lineage |  |
| 14 | regression | 10 | 186 | 29.53–29.540000000000003 | 20.2025 | 9.3275–9.337500000000002 | reconstructed_speech_interval_from_verified_lineage |  |
| 14 | regression | 11 | 187 | 29.58–29.59 | 20.2025 | 9.377499999999998–9.3875 | reconstructed_speech_interval_from_verified_lineage |  |
| 14 | regression | 12 | 187 | 29.58–29.59 | 20.2025 | 9.377499999999998–9.3875 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | classification | 22 | 257 | 183.24–183.25 | 170.347375 | 12.89262500000001–12.902625 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | classification | 7 | 244 | 182.59–182.6 | 170.347375 | 12.242625000000004–12.252624999999995 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | classification | 25 | 260 | 183.39–183.39999999999998 | 170.347375 | 13.042624999999987–13.052624999999978 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | regression | 21 | 256 | 183.19–183.2 | 170.347375 | 12.842624999999998–12.852624999999989 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | regression | 25 | 260 | 183.39–183.39999999999998 | 170.347375 | 13.042624999999987–13.052624999999978 | reconstructed_speech_interval_from_verified_lineage |  |
| 19 | regression | 12 | 249 | 182.84–182.85 | 170.347375 | 12.492625000000004–12.502624999999995 | reconstructed_speech_interval_from_verified_lineage |  |

## Explicit limitations

- Sample 14 positions 34 and 35 share one unaligned/source evidence unit and must not be presented as two independent frames.
- Sample 05 vision remains `index_only`; it was not attempted in this frozen scope.
- Audio/vision mapping outside the frozen 04/14/19 positions remains unchanged and may remain `index_only`.
- A successful reconstructed keyframe is a reproducible navigation frame within the FACET interval, not proof that the official FACET extractor sampled that exact decoded frame.
- T4 is not a Q3 final Gate and does not change the historical 4.6 Gate.

## Reproduction

Use the included `run_t4_raw_evidence_closure.py` from this worktree with the existing T3 private media directory and Attachment4 aligned `videos` directory. No model checkpoint or network download is used.
