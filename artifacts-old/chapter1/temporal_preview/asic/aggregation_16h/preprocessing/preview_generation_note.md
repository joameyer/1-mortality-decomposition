# ASIC Temporal Aggregation Preview: 16h Generation Note

- Alternative aggregation: `16h` completed blocks only.
- Blocked 16h artifacts were derived locally from the standardized ASIC harmonized dynamic table using the same generic upstream blocking contract as the existing 8h artifacts: block membership uses `time_h // block_hours`, `prediction_time_h == block_end_h`, and only structurally completed blocks are kept.
- Cohort logic, proxy within-horizon mortality labels, feature-set boundary, bounded LOCF preprocessing, and baseline model definitions were left unchanged.
- Frozen stay-level split assignments were reused from `/Users/joanameyer/repository/1-mortality-decomposition/artifacts/chapter1/splits/chapter1_stay_split_assignments.csv` rather than being regenerated.
- The preview keeps only the primary feature set for modeling, with dynamic features restricted to median summaries through the existing baseline-selection rule.
- Standardized ASIC source directory: `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized`. Frozen 8h Chapter 1 artifact root: `/Users/joanameyer/repository/1-mortality-decomposition/artifacts/chapter1`.
- This is a narrow preview only. It is not a formal temporal-sensitivity analysis and should not be used by itself to refreeze Chapter 1.