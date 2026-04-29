# ASIC-MIMIC Observation-Process Transferability Note

## Purpose

This note documents whether the ASIC observation-process sensitivity variables transfer to MIMIC.

## ASIC template

The frozen ASIC set contains four group observation indicators, one completeness summary, and four recency variables. The group mapping is HR, BP, respiratory, and oxygenation:

- HR: `heart_rate`
- BP: any of `sbp`, `dbp`, `map`
- Respiratory: `resp_rate`
- Oxygenation: any of `spo2`, `sao2`

## MIMIC transfer

Each group transfers directly at the harmonized concept level.

| Group | ASIC variables | MIMIC harmonized variables | Transfer status | Notes |
|---|---|---|---|---|
| HR | `obs_hr_grp_block`, `tsl_hr_grp_h` | `heart_rate` | `direct_concept_transfer` | Preferred MIMIC source is `icu.chartevents` itemid 220045. |
| BP | `obs_bp_grp_block`, `tsl_bp_grp_h` | `sbp`, `dbp`, `map` | `direct_concept_transfer` | Preferred MIMIC sources are `icu.chartevents` BP itemids 220179, 220050, 220180, 220051, 220181, and 220052. |
| Respiratory | `obs_resp_grp_block`, `tsl_resp_grp_h` | `resp_rate` | `direct_concept_transfer` | Preferred MIMIC sources are total respiratory-rate `icu.chartevents` itemids 220210 and 224690. |
| Oxygenation | `obs_oxy_grp_block`, `tsl_oxy_grp_h` | `spo2`, `sao2` | `direct_concept_transfer` | `spo2` and preferred-source `sao2` both exist; oxygenation is observed if either is observed. |

## Deviations

No material concept-level deviations exist. MIMIC source systems and raw field names differ from ASIC: raw event timestamps are `charttime`, raw numeric values are `valuenum`, and the adapter expresses elapsed time as `minutes_since_admit`. These are source-format differences, not changes to the frozen observation-process concept set.

## Transferability verdict

`direct_concept_transfer`

## Interpretation constraint

Raw observation counts or documentation frequencies are not necessarily numerically comparable across ASIC and MIMIC because source systems and documentation practices differ. The concept-level sensitivity is aligned; raw documentation intensity is not assumed identical.
