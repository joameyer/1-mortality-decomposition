# Chapter 1 ASIC to MIMIC Variable Mapping Note

## Purpose

This note freezes a first durable ASIC -> MIMIC candidate mapping table for the Chapter 1 shared-primary feature contract. The goal is to preserve the smallest semantically defensible MIMIC alignment target for later external-validation preprocessing, without implementing that preprocessing yet.

## Authoritative ASIC Source

The ASIC contract from 5.1.a1 was treated as binding:

- `docs/ch1_asic_feature_contract.md`
- `config/ch1_asic_feature_contract.csv`

The mapping unit in this note is the ASIC **base clinical concept** rather than each derived summary column separately. The shared-primary table therefore maps 31 ASIC base variables and keeps the link back to the usual derived columns through the `asic_example_columns` field in the CSV.

## How Candidate Mappings Were Identified

1. Loaded the completed ASIC feature contract and reduced the 186 shared-primary derived columns to the 31 underlying ASIC base concepts.
2. Inspected the local MIMIC demo schema and dictionaries:
   - `icu/d_items.csv.gz`
   - `hosp/d_labitems.csv.gz`
   - `icu/chartevents.csv.gz`
   - `hosp/labevents.csv.gz`
   - `icu/icustays.csv.gz`
3. For concepts that are not direct single-item measurements, recorded explicit source logic instead of pretending the concept is direct:
   - `pf_ratio` as derived `PaO2 / FiO2_fraction`
   - `vt_per_kg_ibw` as derived `VT / IBW`
4. Used the upstream ASIC harmonization metadata to anchor the target concept semantics and unit expectations where possible:
   - `../icu-data-platform/artifacts/asic_harmonized/dynamic/source_map.csv`
   - `../icu-data-platform/artifacts/asic_harmonized/dynamic/semantic_decisions.csv`
   - `../icu-data-platform/artifacts/asic_harmonized/dynamic/harmonized.csv`
   - `../icu-data-platform/src/icu_data_platform/sources/asic/harmonize/dynamic.py`
5. Computed a lightweight demo coverage estimate from the local MIMIC demo tables only. These counts are intentionally conservative and are not a substitute for full-dataset coverage.

## MIMIC Resources Available During This Task

Available locally:

- MIMIC demo event tables and dictionaries (`chartevents`, `labevents`, `icustays`, `procedureevents`, `admissions`, `patients`, `services`, `d_items`, `d_labitems`)
- The completed ASIC Chapter 1 feature-contract artifacts
- Upstream ASIC harmonization metadata in the sibling `icu-data-platform` repo/artifacts

Not available locally:

- A richer local MIMIC documentation bundle beyond the dictionaries and demo table schemas
- Specimen-level documentation proving arteriality for all blood-gas rows
- A pre-existing local MIMIC variable-mapping note for this exact Chapter 1 contract

Because of that limitation, blood-gas arteriality, some remaining chemistry semantics, and ventilator-derived variables are flagged explicitly rather than smoothed over.


## Unit-informed Revision Pass

A targeted unit-informed revision pass was performed after the first 5.1.a2 draft using the external read-only reference:

- `/Users/joanameyer/repository/icu-mortality-last72h/config/variable_configuration.csv`

This pass did not restart the mapping from scratch. It was limited to tightening unit expectations, writing explicit normalization rules, and separating pure unit mismatch from remaining semantic mismatch.

A filtered Chapter-1-specific derivative of that external reference was saved locally for reproducibility:

- `config/ch1_unit_alignment_reference.csv`

Variables with meaningful unit clarifications in this pass:

- `albumin`: clarified to `g/dL -> dg/L`
- `hemoglobin`: clarified to `g/dL -> mmol/L`
- `bilirubin_total`: clarified to `mg/dL -> µmol/L`
- `creatinine`: clarified to `mg/dL -> µmol/L`
- `urea`: clarified to `mg/dL BUN -> mmol/L urea`
- `wbc`, `platelets`: clarified as a unit-label equivalence (`10x3/µL` vs `10x9/L`) with no numeric rescaling
- `core_temp`, `sao2`, `fio2`, `pao2`, `paco2`, `ph_art`, `lactate_art`: confirmed target-side units more explicitly

Variables that remained semantically weak at the end of the unit pass, before the itemid cross-check below:

- `core_temp`
- `sao2`
- `vt`
- `vt_per_kg_ibw`
- `pf_ratio`
- `ph_art`
- `bicarbonate_art`
- `lactate_art`
- `urea`

Risk changes in this pass:

- `albumin`: `high -> medium` because the prior uncertainty about the ASIC target unit was resolved by the external configuration reference

## Prior-Project Itemid Cross-Check for 5.1.a2

A targeted validation pass was then performed using the external read-only prior-project mapping:

- `/Users/joanameyer/repository/icu-mortality-last72h/config/itemid_finder.yaml`

This YAML was used only as a cross-check artifact, not as ground truth and not as a source to copy wholesale into this repo. The check was limited to:

- `core_temp`
- `sao2`
- `ph_art`
- `lactate_art`
- `vt`
- `pao2`
- `paco2`
- `base_excess_art`
- `bicarbonate_art`
- `fio2`
- `etco2`
- `urea`

The reproducibility artifact for this pass is:

- `config/ch1_itemid_crosscheck_reference.csv`

Prior-project itemids confirmed as supportive of the Chapter 1 mapping:

- `core_temp`: `tem=223762` and `tem_fahrenheit=223761` support the existing generic temperature channels, with the existing temperature-site caution retained.
- `sao2`: `sao=220227` confirms the preferred `Arterial O2 Saturation` chartevents candidate.
- `fio2`: `fio=223835` confirms the `Inspired O2 Fraction` candidate, with percent/fraction handling still explicit because the dictionary unit is blank.
- `pao2`: `oxy=220224` confirms the preferred `Arterial O2 pressure` chartevents candidate.
- `paco2`: `pco=220235` confirms the preferred `Arterial CO2 Pressure` chartevents candidate.
- `urea`: `ure=225624` confirms a MIMIC BUN mirror, but also reinforces that this remains a urea/BUN analyte-conversion proxy rather than a native identical variable.

Prior-project itemids that changed or refined the current mapping:

- `ph_art`: `pha=223830` exists in `d_items` as `PH (Arterial)`. This is more semantically direct than using only broad blood-gas lab pH, so `223830` was promoted to the preferred primary candidate and `50820` remains a conditional secondary lab candidate.
- `vt`: `vt=224684` exists in `d_items` as `Tidal Volume (set)`. Because the Chapter 1 target is setting-oriented, `224684` was promoted to the preferred VT candidate; observed and spontaneous tidal-volume items remain explicit fallback or sensitivity candidates only.

Prior-project itemids rejected or left ambiguous for Chapter 1 alignment:

- `lactate_art`: `lac=225668` exists as chartevents `Lactic Acid`, but the dictionary does not prove arterial or blood-gas context. It was not promoted over blood-gas lactate `50813`/`52442`.
- `base_excess_art`: no corresponding prior-project YAML alias was found; the local `224828`/`50802` mapping was retained.
- `bicarbonate_art`: no corresponding prior-project YAML alias was found; sparse blood-gas bicarbonate `50803` was retained, and serum chemistry bicarbonate should not be silently substituted.
- `etco2`: no corresponding prior-project YAML alias was found; local `228640 EtCO2` was retained with sparse-coverage and blank-unit caution.

Mapping-quality and risk changes in this pass:

- `ph_art`: `mapping_quality` changed from `weak` to `direct` because an explicit arterial pH chartevents item was confirmed. The `risk_flag` remains `high`.
- No `risk_flag` values changed.
- No other `mapping_quality` values changed.

## Follow-Up Itemid Validation for 5.1.a2

A second targeted follow-up checked only five variables against specific supplied itemids and local MIMIC dictionary/event evidence:

- `base_excess_art`
- `bicarbonate_art`
- `etco2`
- `vt`
- `urea`

Accepted as stronger or confirming evidence:

- `base_excess_art`: `224828 Arterial Base Excess` was confirmed in `d_items` as a chartevents lab item and remains the preferred primary candidate. This confirms the existing direct mapping but does not reduce risk because broader use of lab `50802` still depends on arteriality/provenance review.
- `etco2`: `228640 EtCO2` was confirmed in `d_items` as a chartevents routine vital-sign item and remains the preferred primary candidate. Risk remains high because the unit metadata is blank and demo coverage is sparse.
- `vt`: `224684 Tidal Volume (set)` was confirmed as the best setting-oriented Chapter 1 primary candidate. `224685 Tidal Volume (observed)` and `224686 Tidal Volume (spontaneous)` remain explicit fallback or sensitivity candidates only.
- `urea`: `225624 BUN` was confirmed as a charted MIMIC BUN mirror supporting the current BUN/urea-conversion proxy. It does not justify treating `urea` as a direct native urea analyte match.

Rejected as wrong-context candidates for `bicarbonate_art`:

- Serum bicarbonate chart items: `224826 ZHCO3 (serum)` and `227443 HCO3 (serum)` are not arterial bicarbonate measurements.
- APACHE-derived fields: `226759 HCO3ApacheIIValue` and `226760 HCO3Score` are score inputs/components, not raw arterial bicarbonate.
- Treatment/input/ingredient records: `225165 Bicarbonate Base`, `220995 Sodium Bicarbonate 8.4%`, `227533 Sodium Bicarbonate 8.4% (Amp)`, `221211 Sodium Bicarbonate 1,4%`, and `220994 Bicarbonate (ingr)` are not patient bicarbonate measurements.

Mapping-quality and risk changes in this follow-up:

- No `mapping_quality` values changed.
- No `risk_flag` values changed.
- `bicarbonate_art` remains weak/high-risk and close to an operational unavailability concern because the retained best local candidate, `50803 Calculated Bicarbonate, Whole Blood`, is sparse and not explicitly arterial.

## Discussion-Driven Correction on `ph_art` and `core_temp`

A narrow discussion-driven correction pass was applied for only `ph_art` and `core_temp`.

- `ph_art`: confirmed as a direct semantic match when anchored to `223830 PH (Arterial)`. Any remaining caution belongs to coverage/provenance when considering broader secondary `50820` blood-gas pH rows, not to concept mismatch for the preferred arterial pH item.
- `core_temp`: reviewed separately and kept as a non-direct proxy. MIMIC clearly has temperature channels (`223762 Temperature Celsius` and `223761 Temperature Fahrenheit`), but those dictionary labels establish generic temperature availability and unit handling, not proven equivalence to ASIC `Koerperkerntemperatur`. A direct upgrade would require later site/provenance evidence, such as defensible use of `224642 Temperature Site`, showing that the chosen rows truly represent core temperature.

Mapping-quality and risk changes in this correction:

- No `mapping_quality` values changed in this pass: `ph_art` was already `direct`, and `core_temp` remains `proxy`.
- No `risk_flag` values changed.

## High-Risk / Ambiguous Mappings Requiring Later Review

- `core_temp`: the external reference confirms a Celsius target, but the mapping remains a semantic proxy because MIMIC exposes generic temperature channels rather than a clearly core-only temperature concept.
- `sao2`: the external reference confirms percent-to-percent alignment, but arteriality is still not provable for every candidate blood-gas row.
- `fio2`: the external reference clarifies that the aligned contract is percent-based and setting-oriented, but the MIMIC candidate still needs explicit fraction-vs-percent handling and later source-context review.
- `vt`: the external reference clarifies mL and frames the target as a ventilation setting, but the current MIMIC candidate pool still spans observed, set, and spontaneous tidal-volume items.
- `vt_per_kg_ibw`: still requires an explicit later IBW formula choice and height handling; actual body weight should not be substituted silently.
- `pf_ratio`: still has to be derived from aligned `PaO2` and `FiO2` rather than treated as a native direct channel.
- `pao2`, `paco2`, `base_excess_art`: units are already clean, but broader MIMIC coverage often depends on blood-gas rows whose arteriality is not fully demonstrable from the local demo resources alone.
- `ph_art`: the preferred `223830 PH (Arterial)` item is a direct semantic match; the remaining caution is limited to coverage/provenance if broader secondary lab pH rows are used.
- `bicarbonate_art`, `lactate_art`: the cleanest local lab candidates are blood-gas rows, but local resources still do not prove arteriality; `bicarbonate_art` remains especially sparse in the demo.
- `etco2`: direct item exists, but demo coverage is sparse and the local unit field remains blank.
- `crp`: direct candidate exists, but demo coverage is sparse and standard vs high-sensitivity CRP should not be merged silently.
- `urea`: the external reference resolves the target unit as mmol/L but also reinforces that the MIMIC side is a `urea / BUN` construct, so the mapping remains an explicit analyte proxy rather than a raw direct match.

## Variables Currently Marked Proxy / Weak / Unavailable

`proxy`

- `core_temp`
- `urea`

`weak`

- `sao2`
- `vt`
- `vt_per_kg_ibw`
- `pf_ratio`
- `bicarbonate_art`
- `lactate_art`

`unavailable`

- None were marked fully unavailable in this first pass, but `bicarbonate_art` is close to an operational unavailability risk because the semantically clean demo candidate is very sparse and should not be silently replaced by serum bicarbonate.

## MIMIC-Only Infrastructure / Secondary Variables Surfaced During Inspection

These were intentionally kept separate from the shared-primary mapping table.

Likely MIMIC-infrastructure variables:

- `icu.icustays.intime`
- `icu.icustays.outtime`
- `hosp.admissions.deathtime`
- `hosp.admissions.discharge_location`
- `hosp.services.curr_service`
- `icu.procedureevents` ventilation episode codes:
  - `225792 Invasive Ventilation`
  - `225794 Non-invasive Ventilation`
  - `224385 Intubation`
  - `227194 Extubation`

Likely MIMIC-secondary or support variables:

- `icu.chartevents` height items:
  - `226730 Height (cm)`
  - `226707 Height`
- `icu.chartevents` weight items:
  - `226512 Admission Weight (Kg)`
  - `226531 Admission Weight (lbs.)`
  - `224639 Daily Weight`
- `hosp.patients.gender` for IBW derivation support
- `icu.chartevents` ventilator context items:
  - `223848 Ventilator Type`
  - `223849 Ventilator Mode`
  - `229314 Ventilator Mode (Hamilton)`
- `icu.chartevents` `224642 Temperature Site` as a possible later restriction aid for the `core_temp` proxy

## Files and Tables Inspected

ASIC contract and Chapter 1 repo:

- `docs/ch1_asic_feature_contract.md`
- `config/ch1_asic_feature_contract.csv`
- `config/ch1_feature_sets.json`
- `src/chapter1_mortality_decomposition/config.py`
- `src/chapter1_mortality_decomposition/carry_forward.py`
- `src/chapter1_mortality_decomposition/baseline_logistic.py`

Local MIMIC demo resources:

- `mimic-iv-demo/data/icu/d_items.csv.gz`
- `mimic-iv-demo/data/hosp/d_labitems.csv.gz`
- `mimic-iv-demo/data/icu/chartevents.csv.gz`
- `mimic-iv-demo/data/hosp/labevents.csv.gz`
- `mimic-iv-demo/data/icu/icustays.csv.gz`
- `mimic-iv-demo/data/icu/procedureevents.csv.gz`
- `mimic-iv-demo/data/hosp/admissions.csv.gz`
- `mimic-iv-demo/data/hosp/patients.csv.gz`
- `mimic-iv-demo/data/hosp/services.csv.gz`

External unit/configuration reference used in the revision pass:

- `/Users/joanameyer/repository/icu-mortality-last72h/config/variable_configuration.csv`
- `config/ch1_unit_alignment_reference.csv`

Upstream ASIC harmonization metadata used to interpret target semantics and likely unit alignment:

- `../icu-data-platform/artifacts/asic_harmonized/dynamic/source_map.csv`
- `../icu-data-platform/artifacts/asic_harmonized/dynamic/semantic_decisions.csv`
- `../icu-data-platform/artifacts/asic_harmonized/dynamic/harmonized.csv`
- `../icu-data-platform/src/icu_data_platform/sources/asic/harmonize/dynamic.py`
