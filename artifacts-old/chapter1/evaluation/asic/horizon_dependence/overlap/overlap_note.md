# Overlap Note

These outputs use the validated ASIC logistic stay-level hard-case artifact from Package 1.
All local counts and overlap values come from the repository's small synthetic stand-in data and are only for implementation testing.

## Input Artifact

- Hard-case source directory: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression`
- Stay-level source artifact: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`
- Matching key: `stay_id` harmonized from `stay_id_global`, with `hospital_id` checked for consistency across horizons.

## Matched-Denominator Logic

- For each unordered horizon pair, the denominator is the intersection of fatal stay IDs present in both horizons.
- Pairwise overlap metrics do not use all stays and do not use each horizon's raw fatal count as the overlap denominator.
- Directional overlap uses the same matched fatal set for the pair, then divides the intersection by the hard-case count in the source horizon.
- In this local synthetic run, every horizon pair retained the full fatal population on both sides.

| horizon_a | horizon_b | fatal_n_horizon_a | fatal_n_horizon_b | matched_fatal_n |
| --------- | --------- | ----------------- | ----------------- | --------------- |
| 8h        | 16h       | 10                | 10                | 10              |
| 8h        | 24h       | 10                | 10                | 10              |
| 8h        | 48h       | 10                | 10                | 10              |
| 8h        | 72h       | 10                | 10                | 10              |
| 16h       | 24h       | 10                | 10                | 10              |
| 16h       | 48h       | 10                | 10                | 10              |
| 16h       | 72h       | 10                | 10                | 10              |
| 24h       | 48h       | 10                | 10                | 10              |
| 24h       | 72h       | 10                | 10                | 10              |
| 48h       | 72h       | 10                | 10                | 10              |

## Pairwise Overlap Summary

- Mean Jaccard across horizon pairs: `0.690`
- Jaccard range across horizon pairs: `0.429` to `1.000`
- Strongest pair: `8h` vs `16h` with Jaccard `1.000` on matched fatal denominator `10`.
- Weakest pair: `24h` vs `72h` with Jaccard `0.429` on matched fatal denominator `10`.
- Mean directional overlap across ordered pairs: `0.818`

| horizon_a | horizon_b | matched_fatal_n | hard_n_horizon_a | hard_n_horizon_b | intersection_n | union_n | jaccard_index |
| --------- | --------- | --------------- | ---------------- | ---------------- | -------------- | ------- | ------------- |
| 8h        | 16h       | 10              | 4                | 4                | 4              | 4       | 1.000         |
| 8h        | 24h       | 10              | 4                | 4                | 3              | 5       | 0.600         |
| 8h        | 48h       | 10              | 4                | 5                | 4              | 5       | 0.800         |
| 8h        | 72h       | 10              | 4                | 6                | 4              | 6       | 0.667         |
| 16h       | 24h       | 10              | 4                | 4                | 3              | 5       | 0.600         |
| 16h       | 48h       | 10              | 4                | 5                | 4              | 5       | 0.800         |
| 16h       | 72h       | 10              | 4                | 6                | 4              | 6       | 0.667         |
| 24h       | 48h       | 10              | 4                | 5                | 3              | 6       | 0.500         |
| 24h       | 72h       | 10              | 4                | 6                | 3              | 7       | 0.429         |
| 48h       | 72h       | 10              | 5                | 6                | 5              | 6       | 0.833         |

## Persistence Summary

- Fatal-stay union across horizons: `10` stays.
- Persistence table includes separate `available_*`, `fatal_*`, and `hard_case_*` columns so nonfatal or unavailable horizons are not silently mixed with matched fatal denominators.
- Based on the synthetic persistence distribution, hard-case membership looks `more persistent than purely horizon-specific`.

| hard_case_horizon_n | fatal_stay_count | fatal_stay_share |
| ------------------- | ---------------- | ---------------- |
| 0                   | 3                | 0.300            |
| 1                   | 2                | 0.200            |
| 2                   | 1                | 0.100            |
| 3                   | 0                | 0.000            |
| 4                   | 1                | 0.100            |
| 5                   | 3                | 0.300            |

## Caveats

- Later horizons can have smaller stay availability, so matched fatal denominators may be smaller than horizon-specific fatal totals.
- Heatmaps show diagonal self-overlap as `1.00` for readability; the CSV overlap tables only contain cross-horizon pairs.
- Local overlap values are synthetic small-sample outputs and should not be used for substantive interpretation.
