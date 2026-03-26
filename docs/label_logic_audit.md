# Label Logic Audit

## Current Implemented Logic

The current code in [`labels.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/labels.py) implements a **provisional proxy-based within-horizon ICU mortality label**:

- it starts from valid instances that already include `future_window_end_h`
- it joins stay-level `icu_mortality`
- it uses `icu_end_time_proxy_hours` as `event_time_proxy_h`
- label availability requires both non-missing `icu_mortality` and non-missing `icu_end_time_proxy_hours`
- a positive label is assigned only when:
  - `icu_mortality == 1`, and
  - `event_time_proxy_h <= future_window_end_h`
- otherwise the usable label is `0`

## Departure From The Migrated Seed

The migrated seed behavior was different:

- the seed used **terminal ICU mortality reused across horizons**
- it did **not** decide whether death occurred within the prediction horizon
- it explicitly documented that true within-horizon event labels were not available

The current implementation therefore adds a new step:

- it converts a stay-level mortality label plus ICU end-time proxy into a horizon-specific event label

## Why This Is A Scientific Decision

Using `icu_end_time_proxy_hours` as a surrogate event time is not a pure migration cleanup. It changes the scientific meaning of the label:

- it assumes ICU end-time proxy is an acceptable stand-in for death timing when `icu_mortality == 1`
- it changes which instances count as horizon-positive versus horizon-negative
- it can affect performance, calibration, and any downstream interpretation of “poorly captured near-term mortality”

Because of that, this logic should remain explicitly marked as **provisional proxy logic** until the Chapter 1 label definition is scientifically approved.
