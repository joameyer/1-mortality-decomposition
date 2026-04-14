# ASIC ICD-10 Disease-Group Validation Memo

## Implementation

- Disease-group logic module: `src/chapter1_mortality_decomposition/icd10_disease_groups.py`
- Inspection script: `scripts/ch1_asic_icd10_disease_group_inspection.py`
- Local ASIC input root used: `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized`
- Static table inspected: `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv`
- Output directory: `artifacts/chapter1/evaluation/asic/icd10_disease_group_validation`

## Parsing Behavior

- Split `icd10_codes` on comma, strip whitespace, uppercase, remove internal spaces, strip trailing exclamation marks, collapse decimals to normalized 3-character stems, deduplicate within stay, then match on the normalized stem set.
- The frozen first-match hierarchy is applied exactly in this order: surgical / postoperative / trauma-related, respiratory / pulmonary, infection / sepsis non-pulmonary, cardiovascular, neurologic, other / mixed / uncategorized.

## Final Counts

| final_disease_group | stay_count | stay_pct |
| --- | --- | --- |
| surgical / postoperative / trauma-related | 41 | 51.2 |
| respiratory / pulmonary | 30 | 37.5 |
| infection / sepsis non-pulmonary | 1 | 1.2 |
| cardiovascular | 7 | 8.8 |
| neurologic | 1 | 1.2 |
| other / mixed / uncategorized | 0 | 0.0 |

## Ambiguity Burden

| pre_hierarchy_match_bucket | stay_count | stay_pct |
| --- | --- | --- |
| 0 groups | 0 | 0.0 |
| 1 group | 5 | 6.2 |
| 2 groups | 12 | 15.0 |
| 3+ groups | 63 | 78.8 |

- Multi-match burden before hierarchy resolution: `75/80 (94%)`.
- No explicit group match before fallback: `0/80 (0%)`.
- Fallback bucket size after hierarchy assignment: `0/80 (0%)`.

## Common Multi-Match Combinations

| candidate_group_combination | stay_count | stay_pct |
| --- | --- | --- |
| surgical / postoperative / trauma-related + respiratory / pulmonary + infection / sepsis non-pulmonary + cardiovascular + neurologic | 12 | 15.0 |
| surgical / postoperative / trauma-related + respiratory / pulmonary + infection / sepsis non-pulmonary + cardiovascular | 12 | 15.0 |
| respiratory / pulmonary + infection / sepsis non-pulmonary + cardiovascular | 10 | 12.5 |
| respiratory / pulmonary + infection / sepsis non-pulmonary + cardiovascular + neurologic | 9 | 11.2 |
| surgical / postoperative / trauma-related + respiratory / pulmonary + cardiovascular + neurologic | 5 | 6.2 |
| cardiovascular + neurologic | 4 | 5.0 |
| respiratory / pulmonary + cardiovascular | 4 | 5.0 |
| respiratory / pulmonary + cardiovascular + neurologic | 4 | 5.0 |
| surgical / postoperative / trauma-related + respiratory / pulmonary + neurologic | 3 | 3.8 |
| surgical / postoperative / trauma-related + respiratory / pulmonary + infection / sepsis non-pulmonary + neurologic | 2 | 2.5 |

## Edge Cases

- The local artifact still behaves like a diagnosis bag rather than a principal-diagnosis field, so the hierarchy is doing real work.
- `J95` and `I97` are kept inside the earlier surgical/postoperative rule instead of being absorbed into respiratory or cardiovascular matching.
- `N39` is intentionally not used as a standalone infection trigger, and auxiliary stems such as `U80`, `U81`, `Z22`, and `Z29` do not drive assignment on their own.
- Final group distribution is strongly front-loaded: surgical `41/80 (51%)`, respiratory `30/80 (38%)`, infection `1/80 (1%)`, cardiovascular `7/80 (9%)`, neurologic `1/80 (1%)`, other `0/80 (0%)`.

Most frequent stems that do not directly drive any target group in the current rules:

| stem | stays_with_stem | chapter_letter |
| --- | --- | --- |
| Z11 | 46 | Z |
| E87 | 45 | E |
| U69 | 45 | U |
| U99 | 42 | U |
| D62 | 33 | D |
| D68 | 31 | D |
| N17 | 28 | N |
| E11 | 27 | E |
| Z43 | 27 | Z |
| Z29 | 18 | Z |
| D69 | 17 | D |
| E03 | 16 | E |

## Judgment

- Judgment: the implementation is technically ready, but there is a real scientific review point before freezing for HPC: in the local artifact, the bag-of-codes field plus the frozen first-match hierarchy collapses most stays into surgical or respiratory, leaving very small final infection and neurologic buckets. That is not a parser failure, but you should confirm that this degree of hierarchy compression is acceptable for the intended descriptive use.

