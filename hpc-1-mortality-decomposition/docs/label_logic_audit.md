# Label Logic Audit

## Current Implemented Logic

The current code in [`labels.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/labels.py) implements an explicit **proxy within-horizon in-ICU mortality label**:

- it starts from valid instances that already include `future_window_end_h`
- it joins stay-level `icu_mortality`
- it uses `icu_end_time_proxy_hours` as `event_time_proxy_h`
- a positive label is assigned only when:
  - `icu_mortality == 1`
  - `event_time_proxy_h > t`
  - `event_time_proxy_h <= t + H`
- a negative label is assigned only when:
  - `icu_mortality == 0`
  - `event_time_proxy_h >= t + H`
- all other cases remain unlabeled
- unlabeled reasons are summarized explicitly, including survivor-without-full-horizon-observation and non-survivor-proxy-end-not-within-horizon

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

This repository now treats that definition explicitly as the approved ASIC Chapter 1 proxy label, while documenting that it is still a proxy rather than a true event-timed label.
