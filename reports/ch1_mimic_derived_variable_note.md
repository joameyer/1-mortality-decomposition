# Chapter 1 MIMIC Derived Variable Materialization Note

## Purpose

This report documents the narrow materialization check for the frozen derived-only shared-primary variables `pf_ratio` and `vt_per_kg_ibw`.

## Implementation Result

Both derivations were implemented in the completed 8h block-construction stage after preferred-source aggregation and before the MIMIC-to-ASIC adapter consumes `ch1_mimic_blocked_dynamic_features.csv`.

- `pf_ratio = pao2 / (fio2 / 100)`.
- `vt_per_kg_ibw = vt_mL / IBW_kg`, with Devine IBW from height and sex/gender support.

The demo b2 block construction and b3 reused ASIC preprocessing core were rerun after the patch. The existing ASIC preprocessing core ran successfully on the adapted MIMIC inputs.

## Demo QC

- `pf_ratio`: 221/634 completed demo blocks had a block-level derived `last` value; 817/2197 demo primary model-ready rows had `pf_ratio_last` after reused preprocessing.
- `vt_per_kg_ibw`: 208/634 completed demo blocks had a block-level derived `last` value; 1074/2197 demo primary model-ready rows had `vt_per_kg_ibw_last` after reused preprocessing.

## Remaining Limitations

The derivations are block-level translations from frozen support summaries, not timestamp-paired raw-event derivations. Missing PaO2, FiO2, VT, height, or sex/gender support leaves the derived variable missing. Actual body weight is not substituted.

Full-MIMIC row-level/block-level outputs must be regenerated outside the repo before full-run model-ready artifacts reflect this patch.
