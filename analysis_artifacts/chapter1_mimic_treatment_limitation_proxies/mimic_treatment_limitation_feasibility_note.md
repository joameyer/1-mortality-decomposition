# MIMIC Treatment-Limitation / End-of-Life Proxy Feasibility Note

## Scope

This issue inventories structured MIMIC-IV proxies for Chapter 1 interpretation and later hard-case sensitivity. The variables are interpretive threat checks and sensitivity-support variables, not primary mortality-model features.

This issue does not:
- analyze low-predicted fatal cases,
- change the primary cohort,
- change risk-model features,
- use notes/NLP,
- make causal claims.

All findings below refer to structured proxies only. Absence of a structured proxy does not imply absence of treatment limitation.

## Data sources searched

The 5.2a schema/demo scan searched structured sources only: ICU `d_items`, ICU event-table linkage fields for `chartevents`, `datetimeevents`, `procedureevents`, `inputevents`, and `outputevents`; `hosp.poe` and `hosp.poe_detail`; `hosp.admissions`; ICD diagnosis/procedure dictionaries and linked code tables; and `hosp.services`.

The 5.2b full-data count used approved/reviewed candidates from the 5.2a inventory and counted aggregate retained-cohort prevalence from ICU `d_items` linked to `chartevents` and `procedureevents`, ICD diagnosis codes in `diagnoses_icd`, and `admissions.discharge_location`. POE/POE-detail DNR/DNI and palliative consult candidates were found in the schema/demo scan, but `hosp.poe` and `hosp.poe_detail` were unavailable in the selected full MIMIC root, so those candidates could not be counted or timed in 5.2b.

## Candidate proxies found

- `code_status_dnr_dni`: candidates were found in ICU item dictionaries, ICD diagnosis dictionaries, and POE/POE-detail schema/demo values. ICU itemid and ICD candidates were counted in full data. POE/POE-detail DNR/DNI values could not be counted because the full root lacked those tables.
- `comfort_measures_only`: no approved structured candidate was counted. Broad comfort keyword hits were rejected as false positives or insufficiently specific.
- `withdrawal_or_withholding`: no approved structured candidate was counted. Broad withdrawal keyword hits were rejected because they mostly represented substance withdrawal, neurologic withdrawal response, or other non-treatment-limitation concepts.
- `palliative_care`: candidates were found in ICU care-plan items, ICD diagnosis codes, and POE consult order schema/demo values. ICU and ICD candidates were counted; POE consult candidates were unavailable in the full root.
- `hospice`: `admissions.discharge_location` hospice values were counted as discharge/end-of-life context.
- `brain_death_or_organ_donation`: ICU brain-death item and ICD brain-death codes were counted as a separate context domain.
- `ama_or_nonstandard_discharge`: full-data `admissions.discharge_location` was checked for AMA/elopement/left-without-being-seen terms and contributed zero retained Chapter 1 stays.
- `ambiguous_goals_of_care`: family-meeting ICU item candidates were found but not counted because they are nonspecific without value/context review.

## Proxy reliability classification

- `code_status_dnr_dni`: strong; strongest domain. It is a direct structured proxy for documented limitation-positive code status, with timestamped ICU sources and untimed ICD sources. Coverage is incomplete because POE/POE-detail counts were unavailable and comfort/withdrawal/withholding domains were not captured.
- `palliative_care`: moderate descriptive/supporting context. It is common and partly timestamped through ICU event tables, but palliative involvement is not equivalent to DNR/DNI, comfort care, withdrawal, or withholding.
- `hospice`: weak discharge/end-of-life context. It is a hospital discharge disposition marker, not direct ICU treatment limitation.
- `brain_death_or_organ_donation`: moderate separate context domain. It indicates a distinct end-of-life/brain-death pathway and should not be treated as treatment limitation.
- `ama_or_nonstandard_discharge`: weak/checked; zero retained stays counted. It is discharge/care-process context only, not treatment limitation.
- `comfort_measures_only`: not found as an approved structured candidate.
- `withdrawal_or_withholding`: not found as an approved structured candidate.
- `ambiguous_goals_of_care`: rejected/not counted; family-meeting labels are too nonspecific.

## Timing and linkage usability

ICU code-status and palliative care-plan itemids are stay-linked and timestamped through ICU event tables. ICU code-status can support later anchor-aligned sensitivity, but prediction-horizon usability has not been established in this issue and must be tested in later hard-case sensitivity work.

ICD DNR, palliative-care, and brain-death markers are untimed and should be treated as stay-level descriptive context. Hospice and AMA/nonstandard discharge are discharge-context variables. POE order timing could not be assessed because full-root `hosp.poe` and `hosp.poe_detail` tables were unavailable.

## Prevalence summary

5.2b used 10,648 retained Chapter 1 MIMIC stays and 2,210 fatal retained stays by the established `icu_mortality` field.

| Domain | Proxy-positive stays | % of retained stays | Proxy-positive fatal stays | % of fatal stays | Recommended interpretation |
|---|---:|---:|---:|---:|---|
| `code_status_dnr_dni` | 2,638 | 24.8% | 1,386 | 62.7% | Primary structured treatment-limitation proxy |
| `palliative_care` | 1,806 | 17.0% | 1,129 | 51.1% | Descriptive/supporting only |
| `brain_death_or_organ_donation` | 146 | 1.37% | 128 | 5.79% | Separate context domain |
| `hospice` | 231 | 2.17% | 1 | 0.045% | Discharge context only |
| `ama_or_nonstandard_discharge` | 0 | 0.0% | 0 | 0.0% | Checked, absent/zero |
| `comfort_measures_only` | 0 | 0.0% | 0 | 0.0% | No approved structured candidate |
| `withdrawal_or_withholding` | 0 | 0.0% | 0 | 0.0% | No approved structured candidate |
| `ambiguous_goals_of_care` | 0 | 0.0% | 0 | 0.0% | Not counted; too nonspecific |

## Feasibility verdict

Verdict: weakly_testable

- Substantial structured code-status/DNR/DNI signal exists in MIMIC-IV, including 2,638 retained stays and 1,386 fatal retained stays with documented limitation-positive code-status proxies.
- ICU code-status items are stay-linked and timestamped, so later anchor-aligned sensitivity is feasible in principle.
- Palliative markers are common but semantically weaker and must remain descriptive/supporting context only.
- Approved structured comfort-care and withdrawal/withholding proxies were not found.
- ICD markers are untimed, and POE/POE-detail order sources were unavailable in the full root.
- Therefore MIMIC can partially address documented treatment-limitation confounding, but it cannot rule it out and does not support definitive end-of-life sensitivity.

## Recommended use in later Chapter 1 analyses

1. Primary MIMIC proxy sensitivity:
   - Use `code_status_dnr_dni` as the main documented treatment-limitation proxy.
   - Distinguish timestamped ICU code-status items from untimed ICD DNR codes.
   - Use anchor-aligned timing only in later hard-case sensitivity if implemented.

2. Descriptive support:
   - Use `palliative_care` as descriptive/supporting context only.

3. Separate context flag:
   - Use `brain_death_or_organ_donation` separately.

4. Discharge context only:
   - Use `hospice` and `ama_or_nonstandard_discharge` only as discharge/process context.

5. Not recommended:
   - Do not merge all proxies into a crude "EOL" flag for primary interpretation.
   - Do not exclude proxy-positive stays from the primary analysis.
   - Do not treat missing structured proxy as absence of treatment limitation.

## Limitations

- Structured proxies capture documentation, not true care goals.
- Absence of a structured proxy does not imply absence of treatment limitation.
- No approved structured comfort-care, withdrawal, or withholding candidates were counted.
- Palliative, hospice, and AMA domains have different meanings from DNR/DNI and from one another.
- ICD-based markers are untimed.
- POE/POE-detail order sources were unavailable in the full MIMIC root.
- Horizon-specific prediction-anchor alignment was not performed in this issue.
- ASIC and MIMIC availability are asymmetric; MIMIC proxy availability does not solve ASIC's treatment-limitation limitation.

## Files produced

Input artifacts used:
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_inventory_schema_scan.csv`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_schema_scan_note.md`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_full_data_counts.csv`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_timing_summary.csv`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_domain_summary.csv`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_full_data_counts_note.md`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/manifest_full_data_counts.json`
- `docs/context.md`
- `docs/phase1_working_reference.md`
- `docs/phase1_work_packages.md`
- `docs/ch1_mimic_preprocessing_pipeline_manual.md`
- `docs/ch1_mimic_preprocessing_qc.md`

No standalone ASIC treatment-limitation absence note was found in the checked project files.

Produced by 5.2c:
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_feasibility_note.md`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/asic_vs_mimic_treatment_limitation_contrast.md`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/manifest.json`
- `analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_issue_5_2_closure_summary.md`
