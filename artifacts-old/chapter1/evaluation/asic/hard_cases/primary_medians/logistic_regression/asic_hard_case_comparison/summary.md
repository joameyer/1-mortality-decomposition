# ASIC 24h Fatal-Stay Hard-Case Comparison

## Cohort
- Fatal 24h stay-level comparison dataset: `10` stays.
- Low-predicted fatal stays: `4`. Other fatal stays: `6`.
- Hard-case anchor: `asic_logistic_last_eligible_nonfatal_q75_v1` from the saved stay-level hard-case artifact.

## Main Differences
- Low-predicted fatal stays were more common among `asic_UK07` fatal stays (3/4, 75%) than among other fatal stays (0/6, 0%).
- Low-predicted fatal stays were enriched in `respiratory / pulmonary` disease-group assignments (3/4, 75% vs 17%).
- Among the frozen timing and physiologic proxies, PF ratio was higher, creatinine was lower, and PEEP was higher among low-predicted fatal stays.

## Caveat
- This is a bounded descriptive comparison on a small 24h fatal-only slice (`4` vs `6` stays), so the outputs should be read as chapter-oriented structure rather than stable subgroup estimates.
