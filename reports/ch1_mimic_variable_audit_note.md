# Chapter 1 MIMIC Variable Semantic Audit

## Purpose

This report is an aggregated semantic-audit support artifact for Chapter 1 ASIC-to-MIMIC mapping review. It is not preprocessing, cohort construction, 8h block construction, label generation, model fitting, or feature freeze.

All generated outputs are aggregated descriptive summaries only. No patient-level rows, trajectories, timestamps, or extract files are written.

## Data Source

- Audited MIMIC root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1`
- ICU stays in local source: 94458
- Subjects in local source: 65366

The same script can be rerun on demo or full MIMIC by changing `mimic_root` in `config/ch1_mimic_variable_audit.yaml` or passing `--mimic-root`.

## Optional Sources

Core audit tables are required. Auxiliary tables used only for wrong-context checks are optional; if an optional table is unavailable, its candidate rows are retained with `audit_status=skipped_optional_table_missing` and zero counts.

- `ingredientevents` unavailable for `bicarbonate_art` candidate `220994 Bicarbonate (ingr)`; the branch was kept in the audit output as skipped with zero counts.

## Variables Audited

- core_temp
- urea
- sao2
- lactate_art
- vt
- vt_per_kg_ibw
- pf_ratio
- bicarbonate_art
- etco2
- ph_art

`ph_art` was treated as a resolved direct semantic mapping when anchored to `223830 PH (Arterial)`. It is included only as an optional coverage/provenance check, not as an unresolved semantic problem.

## Dominant Demo Sources

- `core_temp`: measurement rows are dominated by 223761 Temperature Fahrenheit (2055040 rows); `224642 Temperature Site` is support metadata, not a measurement source.
- `urea`: accepted proxy sources are dominated by 51006 Urea Nitrogen (4202807 rows).
- `sao2`: accepted/conditional measurement sources are dominated by 50817 Oxygen Saturation (239559 rows).
- `lactate_art`: accepted/conditional measurement sources are dominated by 50813 Lactate (670016 rows).
- `vt`: row counts are dominated by 224685 Tidal Volume (observed) (818760 rows), but the Chapter 1 preferred source remains `224684 Tidal Volume (set)`.
- `bicarbonate_art`: row counts are dominated by wrong-context serum bicarbonate (50882 Bicarbonate (3934240 rows)); the retained blood-gas candidate is 50803 Calculated Bicarbonate, Whole Blood (32994 rows).
- `etco2`: 228640 EtCO2 (159348 rows)
- `ph_art`: 223830 PH (Arterial) (432459 rows) for resolved direct mapping; broader lab pH is coverage/provenance context only.

Dominance here is based on demo row counts only and should not be treated as a full-data freeze decision.

## Semantically Unsafe Or Still Cautious

- core_temp remains semantically unsafe as direct core temperature without site/provenance restriction.
- bicarbonate_art remains unsafe to broaden to serum, APACHE, medication, input, or ingredient bicarbonate candidates.
- urea remains an explicit BUN/urea analyte-conversion proxy, not a native urea match.
- lactate_art and sao2 need source-context caution if broader lab candidates are used.

## Likely Full-Data Dependent Decisions

- Whether sparse demo candidates such as bicarbonate_art and etco2 have enough full-data support.
- Whether broader lab candidates are needed for sao2, lactate_art, pf_ratio inputs, or ph_art coverage.
- Whether temperature-site distributions permit a defensible core-temperature restriction.
- Whether VT set dominates enough to avoid observed/spontaneous fallback rules.

## Output Files

- `reports/ch1_mimic_variable_audit_overview.csv`
- `reports/ch1_mimic_temperature_audit.csv`
- `reports/ch1_mimic_blood_gas_audit.csv`
- `reports/ch1_mimic_vt_audit.csv`
- `reports/ch1_mimic_bicarbonate_audit.csv`
- `reports/ch1_mimic_urea_audit.csv`
- `reports/ch1_mimic_derived_readiness_audit.csv`
