# Chapter 1 Migration Summary

## Classification

### Reusable from the copied seed

- site-level hospital eligibility based on core-vital coverage and ICU mortality availability
- readmission-based first-stay proxy handling
- stay-level exclusion summaries
- block-level feature inventory and Chapter 1 feature-set selection
- model-ready dataset assembly and readiness outputs

### Legacy carryover removed in the standalone repo

- all `icu_data_platform` and `analysis_seed` imports
- upstream path defaults tied to platform internals
- seed-only package naming and structure under `seed/src`
- terminal ICU mortality labels reused unchanged across horizons
- lack of standalone packaging, CLI, and tests

### Still needing follow-up cleanup

- adult-age and mechanical-ventilation-duration stay filters still need a finalized standardized column contract
- external validation and modeling stages remain out of scope for this preprocessing repo

## What This Repo Now Owns

- site-level exclusion logic
- stay-level exclusion logic
- readmission-based first-stay proxy handling
- valid-instance generation from generic 8-hour blocks
- Chapter 1 feature-set configuration
- proxy-based within-horizon label generation using icu_end_time_proxy_hours
- model-ready dataset construction
- Chapter 1 QC and readiness summaries

## What Remains Upstream

- raw extraction
- static and dynamic harmonization
- generic semantic cleaning
- generic stay/block construction
- shared non-Chapter-1 QC

## Important Caveat

This standalone repo emits explicit proxy within-horizon labels using `icu_end_time_proxy_hours` as the event-time surrogate. Exact death timestamps are still unavailable in standardized ASIC artifacts, so these remain proxy labels rather than true event-timed labels.
