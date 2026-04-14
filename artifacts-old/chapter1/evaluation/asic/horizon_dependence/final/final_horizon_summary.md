# Final Horizon Summary

## Inputs Used

| input                               | path                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Saved stay-level hard-case artifact | artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv |
| Package 1 horizon summary           | artifacts/chapter1/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv                             |
| Package 1 foundation note           | artifacts/chapter1/evaluation/asic/horizon_dependence/foundation/artifact_foundation_note.md                     |
| Package 2 pairwise denominators     | artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_denominators.csv                          |
| Package 2 pairwise overlap          | artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv                               |
| Package 2 directional overlap       | artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/directional_overlap.csv                            |
| Package 2 persistence distribution  | artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/persistence_distribution.csv                       |
| Package 2 overlap note              | artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/overlap_note.md                                    |

## Package 3 Outputs

| output                                    | path                                                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| mortality_risk_horizon_comparison.png     | artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png     |
| horizon_interpretation_memo.md            | artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md            |
| final_horizon_summary.md                  | artifacts/chapter1/evaluation/asic/horizon_dependence/final/final_horizon_summary.md                  |
| mortality_risk_horizon_binned_summary.csv | artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_binned_summary.csv |

## Assumptions

- The final figure uses the saved logistic stay-level hard-case artifact as the last-eligible stay-level prediction source for all five horizons.
- For direct visual comparability, the figure uses a common pooled probability binning scheme across horizons while keeping the horizon-specific q75 threshold line from Package 1.
- The memo keeps `24h` as the narrative anchor and `48h` as the main contrast, consistent with the sprint brief.
- The current descriptive label is `change form`, but only as a local implementation-test readout.

## Caveats For Chapter Write-Up

- No consistency mismatches were detected between the plotted horizon-level counts/thresholds and the saved Package 1/2 outputs.
- The mortality-vs-risk figure is descriptive only; it does not reopen or replace the frozen hard-case definition.
- Any chapter narrative should remain bounded to representation-level risk structure under the last-eligible stay design and should avoid causal or subtype claims.
- Current local outputs are implementation-test outputs from synthetic data only and are not scientifically interpretable.
