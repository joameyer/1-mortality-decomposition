# SOFA Feasibility Audit for ASIC Chapter 1 / Sprint 3 / Issue 3.2

## Target Population Used
- Exact artifact: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`.
- Filter applied: `horizon_h == 24` and `label_value == 1`.
- Resulting stay-level fatal comparison population: `10` stays.
- Low-predicted fatal stays (`hard_case_flag == True`): `4`.
- Other fatal stays (`hard_case_flag == False`): `6`.
- This artifact already stores one last eligible prediction instance per stay and horizon; upstream reconstruction is available from `artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_24h/predictions.csv` via `select_last_eligible_stay_points` and `classify_hard_cases_for_horizon` in `src/chapter1_mortality_decomposition/hard_case_definition.py`.

## Component-by-Component Inventory
- Respiratory: `pf_ratio_last` plus `fio2_ventilation_window_active` make a partial standard mapping possible in `7/10 (70%)` fatal stays. `pao2_last` is present in `9/10 (90%)`, `fio2_last` is present in `7/10 (70%)`, and the derived PF ratio is available in `7/10 (70%)`. `spo2_last` is present in `10/10 (100%)`, but an S/F rescue mapping would be nonstandard and should not be introduced for Issue 3.2.
- Coagulation: `platelets_last` is available in `10/10 (100%)`. The raw mapping is standard, but only `3/10 (30%)` are observed in the current 8h block and `7/10 (70%)` depend on 24h LOCF.
- Liver: `bilirubin_total_last` is available in `5/10 (50%)`. Only `2/10 (20%)` are observed in the current block; `3/10 (30%)` rely on 48h LOCF and `5/10 (50%)` remain missing.
- Cardiovascular: `map_last` is available in `9/10 (90%)` and is current-block in `9/10 (90%)`. No vasopressor variables were found in the feature dictionary or the model-ready layer, so standard SOFA cardiovascular scoring cannot be completed.
- CNS: no GCS field was found in the feature dictionary or model-ready dataset (`none`), so the CNS component is absent.
- Renal: `creatinine_last` is available in `8/10 (80%)`. Only `3/10 (30%)` are current-block observations, `5/10 (50%)` depend on 48h LOCF, and no urine-output field was found (`none`).

## Timepoint Alignment Assessment
- The Chapter 1 analysis layer is block-based. Candidate values are 8h-block summaries ending at `prediction_time_h`, not instantaneous bedside measurements.
- Current-block availability across the partially represented SOFA organs is: respiratory `7/10 (70%)`, coagulation `3/10 (30%)`, liver `2/10 (20%)`, cardiovascular MAP `9/10 (90%)`, renal creatinine `3/10 (30%)`.
- Allowing only the repo's existing bounded LOCF windows raises availability to: respiratory `7/10 (70%)`, coagulation `10/10 (100%)`, liver `5/10 (50%)`, cardiovascular MAP `9/10 (90%)`, renal creatinine `8/10 (80%)`.
- Complete-case coverage across those five partially represented organs is `0/10 (0%)` using current-block observations only and `3/10 (30%)` after LOCF.
- Respiratory support status is only indirectly aligned through `fio2_ventilation_window_active`, which flags overlap between the current block and a documented mechanical-ventilation episode.
- Coagulation, liver, and renal coverage are materially dependent on carry-forward from earlier blocks. That makes any score a mixture of current physiology and stale laboratory values rather than a clean same-timepoint severity snapshot.

## Missingness and Completeness Assessment
- Respiratory scorable rows are structured by subgroup: 5/6 other fatal, 2/4 low-predicted fatal.
- Bilirubin availability is strongly structured: 1/6 other fatal, 4/4 low-predicted fatal.
- Creatinine availability is also structured: 4/6 other fatal, 4/4 low-predicted fatal.
- Available-organ complete-case coverage after LOCF is also selective: 1/6 other fatal, 2/4 low-predicted fatal.
- These patterns do not look plausibly missing completely at random. They reflect measurement intensity and case mix, which would bias any complete-case SOFA comparison.
- Component defaulting or imputation would be methodologically unacceptable here. Treating unmeasured GCS, bilirubin, urine output, or vasopressors as normal would mechanically bias the low-predicted fatal comparison.

## Feasibility Classification
- Final classification: `NOT FEASIBLE`.
- Rationale: standard SOFA is blocked by absent CNS, absent vasopressors, and absent urine output. Even a reduced pseudo-SOFA would still depend on nonstandard omissions plus heavy 24-48h carry-forward and would have only `3/10 (30%)` complete cases across the available organs after LOCF.

## Recommendation
- Recommendation: `C. Do not use SOFA in Issue 3.2; proceed with direct organ-support/dysfunction proxies only.`
- A transparent descriptive table of direct proxies is cleaner and more reproducible than introducing a pseudo-SOFA that is missing entire standard domains.

## Implementation Sketch
- Not provided, because feasibility is not clearly positive.
