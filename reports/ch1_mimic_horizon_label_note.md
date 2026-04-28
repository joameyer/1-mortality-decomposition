# Chapter 1 MIMIC Horizon Label Generation Note

## Purpose

This report documents subtask 5.1.b4: verification/export of MIMIC horizon target tables using the frozen ASIC conservative proxy-label scheme.

## Reuse Mode

The existing ASIC label logic was reused directly through the b3 MIMIC-to-ASIC preprocessing adapter. This b4 step verifies the resulting labels and writes horizon-specific target tables; it does not implement a separate MIMIC-native event-time label.

## Semantics

- Positives require ICU mortality with proxy endpoint in `(prediction_time, prediction_time + H]`.
- Negatives require non-ICU-mortality and full horizon observation through `prediction_time + H`.
- Eventual non-survivors outside the current horizon remain unlabeled, not negative.
- Early-discharged survivors without full horizon observation remain unlabeled, not negative.
- Prediction time remains the completed 8h block end.

MIMIC b4 uses the adapter-exposed `icu_end_time_proxy_hours` derived from retained-stay ICU LOS. Stronger MIMIC death timestamps are not substituted into this primary target because doing so would change the frozen ASIC proxy-label semantics.

## Horizons

8h, 16h, 24h, 48h

## Storage

- Preprocessing output root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/preprocessing_outputs`
- Horizon target output root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets`
- Full-MIMIC row-level target tables must remain outside the repo.

## Target Tables

- `8h`: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_8h.csv`
- `16h`: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_16h.csv`
- `24h`: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_24h.csv`
- `48h`: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_48h.csv`

## Verification Status

`pass`

Unlabeled reasons were available from the reused ASIC label output and summarized in `reports/ch1_mimic_horizon_unlabeled_reasons.csv`.

## Deferred Beyond b4

- model fitting
- standard event-time mortality targets
- any redesign of the negative class
