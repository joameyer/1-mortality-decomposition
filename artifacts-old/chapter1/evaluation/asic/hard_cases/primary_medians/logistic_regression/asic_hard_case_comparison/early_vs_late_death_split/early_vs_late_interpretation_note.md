# Early vs Late Fatal Timing Note

- This appendix-only sensitivity reuses the frozen ASIC fatal-stay hard-case artifact anchored to `asic_logistic_last_eligible_nonfatal_q75_v1`.
- Fatal stays are split pragmatically at 48 hours from ICU admission using `icu_end_time_proxy_hours` from the existing Chapter 1 proxy-label workflow: `early ICU death (<=48h)` vs `late ICU death (>48h)`.
- Low-predicted fatal stays were `0/1` (0.0%) among early ICU deaths and `4/9` (44.4%) among late ICU deaths; absolute share difference = `44.4` percentage points.
- The subgroup split is too sparse for a stable descriptive comparison under the conservative implementation rule (>=10 fatal stays per timing group and >=5 stays in each timing x hard-case cell): early ICU death (<=48h) had 1 fatal stays (<10); late ICU death (>48h) had 9 fatal stays (<10); early ICU death (<=48h) had 0 low-predicted fatal stays (<5); early ICU death (<=48h) had 1 other fatal stays (<5); late ICU death (>48h) had 4 low-predicted fatal stays (<5).
- Interpretation: on this run, early-vs-late timing is not informative enough to materially change the existing ASIC hard-case reading and is best treated as decorative.
- This is descriptive only. The late-death group is not a baseline subgroup, so the split should not be given immortal-time-style, causal, or subtype interpretation.
- The 48h split is a pragmatic binary sensitivity choice for interpretability, not evidence of biological death subtypes.
- Admission type was not available in the frozen comparison dataset, so the compact comparison reuses only existing age-group, disease-group, site, timing, creatinine, and PEEP fields from the established hard-case workflow.
