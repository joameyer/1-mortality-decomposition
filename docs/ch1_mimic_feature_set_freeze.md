# Chapter 1 MIMIC Feature Set Freeze

## Purpose

This note freezes the MIMIC Chapter 1 primary aligned feature contract for later external-validation preprocessing. It is a feature-set alignment decision only; it does not build a cohort, construct temporal blocks, create labels, fit models, or claim perfect equivalence between ASIC and MIMIC.

The frozen MIMIC feature set is intended to remain the semantically smallest defensible analogue of the frozen ASIC Chapter 1 design.

## Inputs

This freeze uses the completed Chapter 1 mapping artifacts as authoritative inputs:

- `docs/ch1_asic_feature_contract.md`
- `config/ch1_asic_feature_contract.csv`
- `config/ch1_asic_to_mimic_variable_map.csv`
- `docs/ch1_asic_to_mimic_variable_mapping_note.md`
- `config/ch1_itemid_crosscheck_reference.csv`
- `config/ch1_unit_alignment_reference.csv`
- `reports/ch1_mimic_variable_audit_overview.csv`
- `reports/ch1_mimic_temperature_audit.csv`
- `reports/ch1_mimic_blood_gas_audit.csv`
- `reports/ch1_mimic_vt_audit.csv`
- `reports/ch1_mimic_bicarbonate_audit.csv`
- `reports/ch1_mimic_urea_audit.csv`
- `reports/ch1_mimic_derived_readiness_audit.csv`
- `reports/ch1_mimic_variable_audit_note.md`

The machine-readable freeze summary is:

- `config/ch1_mimic_feature_freeze.csv`

The mapping table also now carries `final_role`, `freeze_decision`, and `freeze_note` columns.

## Final Classification

### Shared-primary direct / near-direct

- `albumin`
- `base_excess_art`
- `bilirubin_total`
- `creatinine`
- `crp`
- `dbp`
- `etco2`
- `fio2`
- `heart_rate`
- `hematocrit`
- `hemoglobin`
- `inr`
- `map`
- `paco2`
- `pao2`
- `peep`
- `ph_art`
- `platelets`
- `ptt`
- `resp_rate`
- `sbp`
- `spo2`
- `wbc`

### Shared-primary proxy retained

- `core_temp`
- `urea`

### Shared-primary retained with accepted asymmetry

- `sao2`
- `lactate_art`
- `vt`

### Shared-primary derived-only

- `pf_ratio`
- `vt_per_kg_ibw`

### MIMIC-secondary

- `bicarbonate_art`

## Key Freeze Decisions

`bicarbonate_art` is demoted from shared-primary to MIMIC-secondary. The retained blood-gas candidate, `50803 Calculated Bicarbonate, Whole Blood`, is semantically closest but sparse and not explicitly arterial in the available resources. Higher-coverage serum bicarbonate, APACHE-derived fields, medication/input records, and ingredient records were explicitly rejected as wrong-context substitutes.

`ph_art` is treated as direct when anchored to `223830 PH (Arterial)`. Any remaining caution is coverage/provenance caution for broader secondary `50820` blood-gas pH rows, not semantic mismatch for the preferred arterial pH item.

`core_temp` remains a proxy. MIMIC has generic temperature channels and a temperature-site support field, but the available evidence does not prove that generic temperature rows are equivalent to ASIC `Koerperkerntemperatur` without later site/provenance restriction.

`urea` remains a proxy. The MIMIC-side sources are Urea Nitrogen/BUN and require explicit analyte/unit conversion to align with ASIC Harnstoff/urea; this is not a native identical analyte match.

`pf_ratio` remains derived-only. It should be computed from accepted PaO2 and FiO2 inputs with explicit FiO2 percent-to-fraction handling rather than treated as a primary measured channel.

`vt_per_kg_ibw` remains derived-only. It should be computed from the accepted VT source plus an explicit IBW derivation from height and sex/gender support variables; actual body weight must not be substituted silently.

## Infrastructure And Support Variables

MIMIC infrastructure variables are preserved for later preprocessing but are not part of the predictive feature set. This includes ICU timing, death/discharge metadata, service type, ventilation episode sources, and height/weight/gender support variables needed for derivations.

This freeze is for external-validation alignment. It preserves known asymmetries and proxy mappings explicitly rather than treating MIMIC and ASIC as perfectly equivalent datasets.
