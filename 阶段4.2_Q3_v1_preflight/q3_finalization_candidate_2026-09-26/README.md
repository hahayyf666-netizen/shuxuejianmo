# Q3 finalization candidate

Original candidate status: `READY_FOR_EXTERNAL_REVIEW`. The first external 4.6 review reported `PASS_WITH_LIMITATIONS` and requested this delivery-semantics correction. External recheck is pending; 4.7 remains on hold. The historical formal 4.6 gate is unchanged.

## Frozen scope

- Input version: `aligned_50`; model: B0; delivery seed: 2029; checkpoint epoch: 2.
- No retraining, architecture/loss/threshold/selection change, or test-driven tuning occurred in this finalization.
- Shapley and conditional IG definitions are unchanged. Explanations describe frozen-model feature-space response, not causal emotion contribution or explanation accuracy.
- Text spans are raw-text evidence only where verified. Audio seconds and visual frames are shown only where T4 lineage passed. Sample 05 vision remains `feature_position_only`; samples 13/16 retain their visual anomaly boundaries.
- T4 remains `T4_COMPLETE_WITH_LIMITATIONS`. Historical T4 files and the historical 4.6 gate were not modified. 4.7 was not entered.
- `reproduction_check.json` means the current frozen implementation reproduced the frozen outputs; it does not prove the exact historical invocation provenance.

## Results

- Attachment4 final CSV: 20 unique rows. Primary-reference raw-evidence coverage: 20/20; classification dominant raw-evidence coverage: 20/20; regression dominant raw-evidence coverage: 19/20. Each count means at least one listed important position for that referenced modality has verified raw evidence; it does not mean all listed positions are mapped. The CSV gives verified/listed counts for audio and vision separately.
- The listed-position denominator is the unique union of classification and regression top positions, which differs from the narrower set attempted by T4. For sample 04, audio is 2/7 listed positions (T4's attempted subset was 2/4) and vision is 3/7 listed positions. Per-position evidence records remain authoritative.
- Valid split: 728 predictions. Accuracy 0.6442307692; Macro-F1 0.6070737874; MAE 0.5964004835; RMSE 0.8137669799; Pearson 0.6290740125. Frozen checkpoint metric comparisons pass at absolute tolerance 1e-6.
- Test consistency: 22/727 direct polarity conflicts; 21 have absolute predicted intensity below 0.2 and 20 have class margin below 0.2. This is descriptive only; no test prediction was changed.
- Frozen reproduction: valid, test, and Attachment4 all pass. Historical runner provenance remains partial.
- Seven paper figures are under `figures/`; sample 04 distinguishes signed support/opposition and uses only T4-verified reconstructed frames.

## Files

- `FINAL_EXPLANATION_SEMANTICS.json`: final names and interpretation rules.
- `q3_final_config.json`: frozen inputs, model, architecture selection record, fixed delivery seed, and explanation settings.
- `results/`: new derived evidence coverage, valid analysis, test consistency, and clean reproduction outputs. Historical formal results remain in their original directories.
- `figures/`: four sample cards and three summary/performance figures.
- `review/Q3_4_6_R1_CANDIDATE.json`: machine-readable readiness evidence. It is not a completed external review.

## Reproduction inputs

The public branch contains the expanded audit handoff, including the frozen test CSV, Attachment4 per-sample results, checkpoint, and scaler. The original approximately 1 GB `aligned_50.pkl` is not stored in GitHub; an authorized local copy is required to rerun valid/test inference. Attachment4 rerun also requires the original 20 aligned PKLs.

Install only the packages in `requirements.txt`. Use paths to the original aligned PKL, Attachment4 directory, frozen `B0_seed2029.pt`, `train_scaler.pt`, explanation contract, C4 text trace, T3 records, and historical output files. The historical predictions and attributions are read-only comparison references.

```powershell
python scripts/build_final_attachment4_results.py
python scripts/run_valid_error_analysis.py --aligned-pkl <aligned_50.pkl> --checkpoint <B0_seed2029.pt> --scaler <train_scaler.pt> --out results --device cpu
python scripts/analyze_test_consistency.py --test-predictions <test_predictions.csv> --out results
python scripts/generate_q3_figures.py --results results --out figures
python scripts/verify_frozen_reproduction.py --aligned-pkl <aligned_50.pkl> --attachment-dir <attachment4_aligned_pkls> --checkpoint <B0_seed2029.pt> --scaler <train_scaler.pt> --contract <attachment4_explanation_contract.json> --text-trace <text_row_trace_564.csv> --t3-records <t3_media_origin_records.json> --historical-test-predictions <test_predictions.csv> --historical-test-summary <test_evaluation_summary.json> --historical-attachment-results <formal_attachment4_results> --out reproduction_tmp
```

Run the final reproduction command only with an empty `reproduction_tmp/`. It writes fresh outputs there and a comparison report under `results/`; it does not write into historical result folders. `reproduction_tmp/` is a local scratch directory excluded from this commit; the comparison outcome is committed as `results/reproduction_check.json`. The candidate `.gitattributes` preserves generated `results/` files byte for byte so recorded CSV hashes also match the Git blobs. Do not use test metrics for model or threshold choices.
