# MIMIC Treatment-Limitation / End-of-Life Proxy Schema Scan Note

## Scope

This is schema/demo/dictionary discovery only for candidate structured proxies. It makes no full-cohort prevalence claims, performs no low-predicted fatal-case analysis, does not inspect notes/NLP, and does not alter cohort definitions or primary risk-model features. All variables named here are preliminary proxy candidates or rejected keyword hits for review.

Run mode: MIMIC-IV demo structured rows and dictionaries were inspected.

## Sources inspected

- `icu.chartevents`: available; header/schema only; columns: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning. header inspected for d_items linkage/timing fields
- `icu.datetimeevents`: available; header/schema only; columns: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valueuom, warning. header inspected for d_items linkage/timing fields
- `icu.procedureevents`: available; header/schema only; columns: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, value, valueuom, location, locationcategory, orderid, linkorderid, ordercategoryname, ordercategorydescription, patientweight, isopenbag, continueinnextdept, statusdescription, ORIGINALAMOUNT, ORIGINALRATE. header inspected for d_items linkage/timing fields
- `icu.inputevents`: available; header/schema only; columns: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, amount, amountuom, rate, rateuom, orderid, linkorderid, ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription, patientweight, totalamount, totalamountuom, isopenbag, continueinnextdept, statusdescription, originalamount, originalrate. header inspected for d_items linkage/timing fields
- `icu.outputevents`: available; header/schema only; columns: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valueuom. header inspected for d_items linkage/timing fields
- `hosp.poe`: available; rows scanned; columns: poe_id, poe_seq, subject_id, hadm_id, ordertime, order_type, order_subtype, transaction_type, discontinue_of_poe_id, discontinued_by_poe_id, order_provider_id, order_status. rows scanned
- `hosp.poe_detail`: available; rows scanned; columns: poe_id, poe_seq, subject_id, field_name, field_value. rows scanned
- `icu.d_items`: available; rows scanned; columns: itemid, label, abbreviation, linksto, category, unitname, param_type, lownormalvalue, highnormalvalue. rows scanned
- `hosp.admissions`: available; rows scanned; columns: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag. rows scanned
- `hosp.diagnoses_icd`: available; header/schema only; columns: subject_id, hadm_id, seq_num, icd_code, icd_version. header inspected only; no patient-level ICD rows scanned
- `hosp.procedures_icd`: available; header/schema only; columns: subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version. header inspected only; no patient-level ICD rows scanned
- `hosp.d_icd_diagnoses`: available; rows scanned; columns: icd_code, icd_version, long_title. rows scanned
- `hosp.d_icd_procedures`: available; rows scanned; columns: icd_code, icd_version, long_title. rows scanned
- `hosp.services`: available; rows scanned; columns: subject_id, hadm_id, transfertime, prev_service, curr_service. rows scanned

## Candidate proxy domains found

- `code_status_dnr_dni`: 10 candidate inventory row(s). Examples: Code Status (Intubation) | Code Status (Intubation) | Intubation; Code Status | Code Status | General; Code Status. | Code Status. | MD Progress Note; Code status | DNAR (DO NOT attempt resuscitation for cardiac arrest); Code status | Do not resuscitate (DNR/DNI)
- `comfort_measures_only`: no candidate inventory rows found.
- `withdrawal_or_withholding`: no candidate inventory rows found.
- `palliative_care`: 8 candidate inventory row(s). Examples: Consults | Palliative Care | Inactive; Consults | Palliative Care/Ethics Support | Inactive; Encounter for palliative care; Palliative Care NCP - Expected outcomes | Palliative Care NCP - Expected outcomes | Care Plans; Palliative Care NCP - Interventions | Palliative Care NCP - Interventions | Care Plans
- `hospice`: 1 candidate inventory row(s). Examples: HOSPICE
- `brain_death_or_organ_donation`: 3 candidate inventory row(s). Examples: Brain Death | Brain Death | 3-Significant Events; Brain death
- `ama_or_nonstandard_discharge`: no candidate inventory rows found.
- `ambiguous_goals_of_care`: 3 candidate inventory row(s). Examples: Family Meeting | Family Meeting | Restraint/Support Systems; Family meeting attempted, unable | Family meeting attempted, unable | 7-Communication; Family meeting held | Family meeting held | 7-Communication
- `reject_false_positive`: 64 candidate inventory row(s). Examples: Adjustment reaction with withdrawal; Alcohol dependence with withdrawal; Alcohol dependence with withdrawal delirium; Alcohol dependence with withdrawal with perceptual disturbance; Alcohol dependence with withdrawal, uncomplicated

## Sources not available or not inspected

- `icu.chartevents`: not row-scanned; header inspected for d_items linkage/timing fields.
- `icu.datetimeevents`: not row-scanned; header inspected for d_items linkage/timing fields.
- `icu.procedureevents`: not row-scanned; header inspected for d_items linkage/timing fields.
- `icu.inputevents`: not row-scanned; header inspected for d_items linkage/timing fields.
- `icu.outputevents`: not row-scanned; header inspected for d_items linkage/timing fields.
- `hosp.diagnoses_icd`: not row-scanned; header inspected only; no patient-level ICD rows scanned.
- `hosp.procedures_icd`: not row-scanned; header inspected only; no patient-level ICD rows scanned.

## Preliminary classification

- `strong`: 4 candidate inventory row(s).
- `moderate`: 15 candidate inventory row(s).
- `weak`: 6 candidate inventory row(s).
- `reject`: 64 candidate inventory row(s).

## Recommended candidate list for full-data aggregation

- `brain_death_or_organ_donation` from `hosp.d_icd_diagnoses` `long_title` `ICD10:G9382`: Brain death (include_for_5_2b; full_data_timing_check).
- `brain_death_or_organ_donation` from `hosp.d_icd_diagnoses` `long_title` `ICD9:34882`: Brain death (include_for_5_2b; full_data_timing_check).
- `brain_death_or_organ_donation` from `icu.d_items` `label+abbreviation+category` `225819`: Brain Death | Brain Death | 3-Significant Events (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `hosp.d_icd_diagnoses` `long_title` `ICD10:Z66`: Do not resuscitate (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `hosp.d_icd_diagnoses` `long_title` `ICD9:V4986`: Do not resuscitate status (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `hosp.poe` `order_type+order_subtype+order_status` ``: General Care | Code status | Inactive (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `hosp.poe_detail` `field_name+field_value` ``: Code status | DNAR (DO NOT attempt resuscitation for cardiac arrest) (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `hosp.poe_detail` `field_name+field_value` ``: Code status | Do not resuscitate (DNR/DNI) (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `icu.d_items` `label+abbreviation+category` `223758`: Code Status | Code Status | General (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `icu.d_items` `label+abbreviation+category` `228687`: Code Status. | Code Status. | MD Progress Note (include_for_5_2b; full_data_timing_check).
- `code_status_dnr_dni` from `icu.d_items` `label+abbreviation+category` `229784`: Code Status (Intubation) | Code Status (Intubation) | Intubation (include_for_5_2b; full_data_timing_check).
- `hospice` from `hosp.admissions` `discharge_location` ``: HOSPICE (include_descriptive_only_for_5_2b; full_data_prevalence_count).
- `palliative_care` from `hosp.d_icd_diagnoses` `long_title` `ICD10:Z515`: Encounter for palliative care (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `hosp.d_icd_diagnoses` `long_title` `ICD9:V667`: Encounter for palliative care (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `hosp.poe` `order_type+order_subtype+order_status` ``: Consults | Palliative Care | Inactive (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `hosp.poe` `order_type+order_subtype+order_status` ``: Consults | Palliative Care/Ethics Support | Inactive (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `icu.d_items` `label+abbreviation+category` `229150`: Palliative Care NCP - Expected outcomes | Palliative Care NCP - Expected outcomes | Care Plans (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `icu.d_items` `label+abbreviation+category` `229152`: Palliative Care NCP - Interventions | Palliative Care NCP - Interventions | Care Plans (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `icu.d_items` `label+abbreviation+category` `229154`: Palliative Care NCP - Plan revised | Palliative Care NCP - Plan revised | Care Plans (include_descriptive_only_for_5_2b; full_data_timing_check).
- `palliative_care` from `icu.d_items` `label+abbreviation+category` `229155`: Palliative Care NCP - Problem resolved | Palliative Care NCP - Problem resolved | Care Plans (include_descriptive_only_for_5_2b; full_data_timing_check).

## Warnings for review

- `ambiguous_goals_of_care` from `icu.d_items` `label+abbreviation+category`: Family Meeting | Family Meeting | Restraint/Support Systems. Reason: Goals-of-care/family-meeting language is nonspecific without value context.
- `ambiguous_goals_of_care` from `icu.d_items` `label+abbreviation+category`: Family meeting held | Family meeting held | 7-Communication. Reason: Goals-of-care/family-meeting language is nonspecific without value context.
- `ambiguous_goals_of_care` from `icu.d_items` `label+abbreviation+category`: Family meeting attempted, unable | Family meeting attempted, unable | 7-Communication. Reason: Goals-of-care/family-meeting language is nonspecific without value context.
- `code_status_dnr_dni` from `hosp.poe_detail` `field_name+field_value`: Code status | Full code  (attempt resuscitation). Reason: Full-code value identifies a code-status source but is not treatment-limitation positive.
- `code_status_dnr_dni` from `hosp.poe_detail` `field_name+field_value`: Code status | Resuscitate (Full code). Reason: Full-code value identifies a code-status source but is not treatment-limitation positive.
- `reject_false_positive` from `hosp.admissions` `deathtime`: deathtime. Reason: Outcome/death-discharge marker, not a treatment-limitation proxy.
- `reject_false_positive` from `hosp.admissions` `hospital_expire_flag`: hospital_expire_flag. Reason: Outcome/death-discharge marker, not a treatment-limitation proxy.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol dependence with withdrawal, uncomplicated. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol dependence with withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol dependence with withdrawal with perceptual disturbance. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol dependence with withdrawal, unspecified. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Opioid dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Opioid use, unspecified with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Cannabis dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Cannabis use, unspecified with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic dependence with withdrawal, uncomplicated. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic dependence with withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic dependence with withdrawal with perceptual disturbance. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic dependence with withdrawal, unspecified. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic use, unspecified with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic use, unspecified with withdrawal, uncomplicated. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic use, unspecified with withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic use, unspecified with withdrawal with perceptual disturbances. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Sedative, hypnotic or anxiolytic use, unspecified with withdrawal, unspecified. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Cocaine dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other stimulant dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other stimulant use, unspecified with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Nicotine dependence unspecified, with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Nicotine dependence, cigarettes, with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Nicotine dependence, chewing tobacco, with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Nicotine dependence, other tobacco product, with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance dependence with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance dependence with withdrawal, uncomplicated. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance dependence with withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance dependence with withdrawal with perceptual disturbance. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance dependence with withdrawal, unspecified. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance use, unspecified with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance use, unspecified with withdrawal, uncomplicated. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance use, unspecified with withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance use, unspecified with withdrawal with perceptual disturbance. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Other psychoactive substance use, unspecified with withdrawal, unspecified. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort, right eye. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort, left eye. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort, bilateral. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort, unspecified. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Neonatal withdrawal symptoms from maternal use of drugs of addiction. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Withdrawal symptoms from therapeutic use of drugs in newborn. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal, unspecified time. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal, in the field [EMT or ambulance]. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal, at arrival to emergency department. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal, at hospital admission. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Coma scale, best motor response, flexion withdrawal, 24 hours or more after hospital admission. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol withdrawal delirium. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Alcohol withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Drug withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Adjustment reaction with withdrawal. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Visual discomfort. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `hosp.d_icd_diagnoses` `long_title`: Drug withdrawal syndrome in newborn. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `hosp.d_icd_procedures` `long_title`: Insertion of catheter into spinal canal for infusion of therapeutic or palliative substances. Reason: Palliative keyword refers to medication/substance purpose in a procedure label, not palliative-care service or goals-of-care context.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Status and Comfort (Behavioral) | Status and Comfort (Behavioral) | Restraint/Support Systems. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Status and Comfort | Status and Comfort | Restraint/Support Systems. Reason: Comfort keyword lacks comfort-care/comfort-measures/CMO context.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Substance withdrawal NCP - Expected outcomes | Substance withdrawal NCP - Expected outcomes | Care Plans. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Substance withdrawal NCP - Interventions | Substance withdrawal NCP - Interventions | Care Plans. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Substance withdrawal NCP - Plan revised | Substance withdrawal NCP - Plan revised | Care Plans. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.
- `reject_false_positive` from `icu.d_items` `label+abbreviation+category`: Substance withdrawal NCP - Problem resolved | Substance withdrawal NCP - Problem resolved | Care Plans. Reason: Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.

AMA, elopement, and left-against-medical-advice markers are discharge-process or care-process context flags only. They are not treatment-limitation or end-of-life proxies. Palliative care is not equivalent to DNR/DNI, withdrawal, or withholding. Hospice discharge is discharge/end-of-life context, not direct ICU treatment limitation unless stronger structured evidence is found in a later full-data timing/value check. Absence of a structured marker must not be interpreted as absence of treatment limitation.

## Files produced

- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_inventory_schema_scan.csv`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_schema_scan_note.md`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/manifest_schema_scan.json`
