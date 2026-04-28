# Chapter 1 MIMIC Horizon Label Generation

## Purpose

This note documents subtask 5.1.b4: generation and verification of MIMIC horizon targets for `8h`, `16h`, `24h`, and `48h`.

MIMIC b4 preserves the frozen ASIC Chapter 1 conservative proxy-label scheme. It does not switch to standard event-time mortality labels and does not redesign the positive, negative, or unlabeled classes.

## Implementation Mode

The existing ASIC label logic was reused directly after the b3 MIMIC-to-ASIC preprocessing adapter. The source of truth remains:

- `src/chapter1_mortality_decomposition/labels.py`
- `docs/label_logic_audit.md`

The b4 code is a thin verification/export layer:

- `src/chapter1_mortality_decomposition/mimic_horizon_labels.py`
- `scripts/run_mimic_horizon_labels.py`
- `config/ch1_mimic_horizon_labels.yaml`

It reads the reused preprocessing output `labels/chapter1_proxy_horizon_labels.csv`, verifies the frozen semantics against `cohort/chapter1_retained_stay_table.csv`, and writes horizon-specific target tables outside the repo for full MIMIC.

## Frozen Label Semantics

For a horizon `H`, with prediction time equal to the completed-block end:

```text
future_window_end_h = prediction_time_h + H
```

A row is positive only if:

```text
icu_mortality == 1
and event_time_proxy_h > prediction_time_h
and event_time_proxy_h <= future_window_end_h
```

A row is negative only if:

```text
icu_mortality == 0
and event_time_proxy_h >= future_window_end_h
```

All other rows remain unlabeled.

This means:

- negatives remain certain horizon survivors only
- eventual non-survivors outside the current horizon remain unlabeled, not negative
- early-discharged survivors without full horizon observation remain unlabeled, not negative
- prediction time remains the completed 8h block end

## MIMIC Timing Translation

The b3 adapter exposes:

```text
icu_end_time_proxy_hours = icu_los_hours
```

using retained-stay `intime` and `outtime`. B4 therefore uses the same proxy endpoint as the reused ASIC label logic.

MIMIC has stronger timing fields, including `admissions.deathtime`, but these are not substituted into the primary b4 label. Using true death timestamps would change the frozen ASIC proxy-label semantics and would create a different target.

## Horizons

Generated and verified horizons:

- `8h`
- `16h`
- `24h`
- `48h`

The existing preprocessing core may also generate other configured horizons internally, but b4 exports and verifies only the four frozen MIMIC target horizons above.

## Outputs

Safe repo-local reports:

- `reports/ch1_mimic_horizon_label_summary.csv`
- `reports/ch1_mimic_horizon_unlabeled_reasons.csv`
- `reports/ch1_mimic_horizon_label_note.md`

Full-MIMIC row-level target tables were written outside the repo:

- `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_8h.csv`
- `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_16h.csv`
- `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_24h.csv`
- `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets/ch1_mimic_proxy_horizon_targets_48h.csv`

## Verification

The full-MIMIC b4 run passed semantic verification for all four horizons.

The verifier checks:

- positive rows satisfy the conservative positive rule
- negative rows satisfy the conservative negative rule
- rows that should be labelable are not left unlabeled
- eventual non-survivors outside the current horizon are not labeled negative
- early-discharged survivors are not labeled negative

The full-MIMIC summary is stored in `reports/ch1_mimic_horizon_label_summary.csv`.

## Unlabeled Reasons

Explicit unlabeled reasons are available from the reused ASIC label output and are summarized in:

```text
reports/ch1_mimic_horizon_unlabeled_reasons.csv
```

Observed reason families in the full-MIMIC b4 verification:

- `non_survivor_proxy_end_not_within_horizon`
- `survivor_without_full_horizon_observation`

## Deferred Beyond b4

The following remain outside b4:

- model fitting
- model-ready evaluation
- standard event-time mortality target construction
- any redesign of the negative class
- any sensitivity analysis comparing proxy labels with true MIMIC event-time labels
