# Artifact Foundation Note

Local outputs in this directory come from the repository's small synthetic stand-in data.
They verify implementation and artifact contracts only; they are not scientifically interpretable.

## Input Files Used

- Hard-case manifest: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/run_manifest.json`
- Saved stay-level hard-case artifact: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`
- Saved horizon summary artifact: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- Hard-case rule from manifest: `asic_logistic_last_eligible_nonfatal_q75_v1`
- Horizons inspected: 8h, 16h, 24h, 48h, 72h

## Shared Stay-Level Hard-Case Artifact

- Path: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`
- Format: `csv`
- Columns: `stay_id_global, hospital_id, horizon_h, split, label_value, instance_id, block_index, prediction_time_h, predicted_probability, model_name, nonfatal_q75_threshold, hard_case_flag, hard_case_rule`
- Stable stay identifier field: `stay_id_global`
- Fatal/nonfatal label field: `label_value`
- Predicted-risk field: `predicted_probability`
- Hard-case flag field: `hard_case_flag`
- Horizon-specific threshold field: `nonfatal_q75_threshold`
- Schema status: one combined harmonized stay-level file spans all five horizons via `horizon_h`.

## Horizon-Specific Prediction Inputs

| horizon | prediction_path                                                                                   | format | stay_id_field  | label_field | predicted_risk_field  | hard_case_flag_field             | threshold_field                                     |
| ------- | ------------------------------------------------------------------------------------------------- | ------ | -------------- | ----------- | --------------------- | -------------------------------- | --------------------------------------------------- |
| 8h      | artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_8h/predictions.csv  | csv    | stay_id_global | label_value | predicted_probability | derived_from_stay_level_artifact | nonfatal_q75_threshold_saved_in_stay_level_artifact |
| 16h     | artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_16h/predictions.csv | csv    | stay_id_global | label_value | predicted_probability | derived_from_stay_level_artifact | nonfatal_q75_threshold_saved_in_stay_level_artifact |
| 24h     | artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_24h/predictions.csv | csv    | stay_id_global | label_value | predicted_probability | derived_from_stay_level_artifact | nonfatal_q75_threshold_saved_in_stay_level_artifact |
| 48h     | artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_48h/predictions.csv | csv    | stay_id_global | label_value | predicted_probability | derived_from_stay_level_artifact | nonfatal_q75_threshold_saved_in_stay_level_artifact |
| 72h     | artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_72h/predictions.csv | csv    | stay_id_global | label_value | predicted_probability | derived_from_stay_level_artifact | nonfatal_q75_threshold_saved_in_stay_level_artifact |

Prediction artifact columns by horizon:

| horizon | columns                                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 8h      | instance_id, stay_id_global, hospital_id, block_index, prediction_time_h, horizon_h, split, label_value, predicted_probability, model_name |
| 16h     | instance_id, stay_id_global, hospital_id, block_index, prediction_time_h, horizon_h, split, label_value, predicted_probability, model_name |
| 24h     | instance_id, stay_id_global, hospital_id, block_index, prediction_time_h, horizon_h, split, label_value, predicted_probability, model_name |
| 48h     | instance_id, stay_id_global, hospital_id, block_index, prediction_time_h, horizon_h, split, label_value, predicted_probability, model_name |
| 72h     | instance_id, stay_id_global, hospital_id, block_index, prediction_time_h, horizon_h, split, label_value, predicted_probability, model_name |

## Schema Consistency Assessment

- Prediction schemas identical across horizons: `True`
- Stay identifier available across all horizons: `True` (`stay_id_global` is non-missing and unique within each `stay_id_global`/`horizon_h` pair).
- `stay_id_global` maps to exactly one `hospital_id`: `True`
- Any horizon missing fields needed for later overlap analysis: `False`
- Saved summary cross-check: Derived counts and thresholds exactly matched the saved `horizon_hard_case_summary.csv` artifact.

## Per-Horizon Stay Coverage

| horizon | stay_rows | unique_stays | nonfatal_last_n | fatal_last_n | missing_vs_8h                                  |
| ------- | --------- | ------------ | --------------- | ------------ | ---------------------------------------------- |
| 8h      | 35        | 35           | 25              | 10           | none                                           |
| 16h     | 35        | 35           | 25              | 10           | none                                           |
| 24h     | 35        | 35           | 25              | 10           | none                                           |
| 48h     | 34        | 34           | 24              | 10           | asic_UK08_9993                                 |
| 72h     | 32        | 32           | 22              | 10           | asic_UK02_9996, asic_UK07_9996, asic_UK08_9993 |

## Known True-Data Reference Comparison

All local values differ from the known HPC summary, which is expected because this repository uses small synthetic stand-in data.

| horizon | local_nonfatal_last_n | target_nonfatal_last_n | local_fatal_last_n | target_fatal_last_n | local_q75 | target_q75 | local_hard_case_n | target_hard_case_n |
| ------- | --------------------- | ---------------------- | ------------------ | ------------------- | --------- | ---------- | ----------------- | ------------------ |
| 8h      | 25                    | 4713                   | 10                 | 1639                | 0.008151  | 0.004425   | 4                 | 351                |
| 16h     | 25                    | 4713                   | 10                 | 1670                | 0.027625  | 0.009295   | 4                 | 342                |
| 24h     | 25                    | 4696                   | 10                 | 1682                | 0.040880  | 0.014598   | 4                 | 346                |
| 48h     | 24                    | 4542                   | 10                 | 1697                | 0.110249  | 0.032415   | 5                 | 352                |
| 72h     | 22                    | 4326                   | 10                 | 1704                | 0.163779  | 0.052678   | 6                 | 364                |

## Mismatches And Ambiguities

- No mismatches were found between the saved stay-level artifact and the saved horizon summary artifact.
- Known-target difference: 8h nonfatal_last_n local=25 vs known_target=4713
- Known-target difference: 8h fatal_last_n local=10 vs known_target=1639
- Known-target difference: 8h nonfatal_q75_threshold local=0.008151376724 vs known_target=0.004425
- Known-target difference: 8h hard_case_n local=4 vs known_target=351
- Known-target difference: 16h nonfatal_last_n local=25 vs known_target=4713
- Known-target difference: 16h fatal_last_n local=10 vs known_target=1670
- Known-target difference: 16h nonfatal_q75_threshold local=0.027625149892 vs known_target=0.009295
- Known-target difference: 16h hard_case_n local=4 vs known_target=342
- Known-target difference: 24h nonfatal_last_n local=25 vs known_target=4696
- Known-target difference: 24h fatal_last_n local=10 vs known_target=1682
- Known-target difference: 24h nonfatal_q75_threshold local=0.040880016557 vs known_target=0.014598
- Known-target difference: 24h hard_case_n local=4 vs known_target=346
- Known-target difference: 48h nonfatal_last_n local=24 vs known_target=4542
- Known-target difference: 48h fatal_last_n local=10 vs known_target=1697
- Known-target difference: 48h nonfatal_q75_threshold local=0.110248877162 vs known_target=0.032415
- Known-target difference: 48h hard_case_n local=5 vs known_target=352
- Known-target difference: 72h nonfatal_last_n local=22 vs known_target=4326
- Known-target difference: 72h fatal_last_n local=10 vs known_target=1704
- Known-target difference: 72h nonfatal_q75_threshold local=0.163779487570 vs known_target=0.052678
- Known-target difference: 72h hard_case_n local=6 vs known_target=364

## Package 2 Readiness

Package 2 can proceed cleanly: stable stay matching is available via `stay_id_global`, with `hospital_id` available as a defensive secondary key. Later overlap work still needs to respect the horizon-specific coverage shrinkage at 48h and 72h.

Local numeric values in this note are synthetic test outputs only and should not be substantively interpreted.
