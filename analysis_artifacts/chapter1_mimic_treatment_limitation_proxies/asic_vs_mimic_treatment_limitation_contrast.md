# ASIC vs MIMIC Treatment-Limitation Proxy Contrast

## Purpose

This note contrasts dataset support for treatment-limitation, end-of-life, and discharge-context sensitivity in Chapter 1. It is intended to support bounded interpretation of later ASIC-vs-MIMIC hard-case comparisons.

## ASIC status

Current project framing in `docs/context.md` states that treatment-limitation / end-of-life confounding remains unresolved and is still the largest open interpretation limit in ASIC. The same document says to keep treatment-limitation absence explicit rather than implying it was solved empirically.

`docs/phase1_working_reference.md` names treatment-limitation confounding as a risk: a substantial share of low-predicted mortality could reflect care-limitation processes rather than sudden physiologic collapse. It also states that treatment-limitation/end-of-life variables should be inventoried and that absence of such variables must be documented explicitly.

No separate ASIC treatment-limitation absence memo was found in the checked project docs/artifacts. ASIC direct treatment-limitation/end-of-life sensitivity therefore remains unresolved or not directly testable from the current project record.

## MIMIC status

MIMIC-IV has substantial structured code-status/DNR/DNI proxy availability. In the retained Chapter 1 MIMIC cohort, `code_status_dnr_dni` proxies were present in 2,638 of 10,648 stays and 1,386 of 2,210 fatal stays. ICU code-status sources are timestamped and stay-linked, while ICD DNR codes are useful as untimed stay-level descriptive context.

MIMIC also has palliative descriptive markers, brain-death/organ-donation context, and hospice/AMA discharge-context checks. Palliative-care markers are common but are not equivalent to treatment limitation. Brain death / organ donation is a separate context domain. Hospice discharge is discharge/end-of-life context, not direct ICU treatment limitation. AMA/nonstandard discharge was checked and contributed zero retained stays.

No approved structured comfort-care or withdrawal/withholding candidates were counted. POE/POE-detail candidates from the schema/demo scan could not be counted in the full root because `hosp.poe` and `hosp.poe_detail` were unavailable. The final MIMIC verdict is `weakly_testable`.

## Dataset asymmetry

MIMIC has stronger structured code-status proxy support than ASIC. MIMIC can partially test documented treatment-limitation confounding through code-status/DNR/DNI proxies and supporting context variables. This does not solve ASIC's treatment-limitation limitation, because ASIC lacks equivalent documented structured support in the current project record.

ASIC-vs-MIMIC comparisons must therefore treat treatment-limitation sensitivity as asymmetric: MIMIC can support a bounded proxy sensitivity, while ASIC remains limited by explicit unresolved treatment-limitation/end-of-life confounding.

## Contrast table

| Dimension | ASIC | MIMIC | Interpretation |
|---|---|---|---|
| Direct DNR/DNI proxy | Not directly testable in current project record | Present through ICU code-status items and ICD DNR codes | MIMIC has usable documented code-status proxy support; ASIC remains limited |
| Comfort-care proxy | Not documented as available | No approved structured candidate counted | Neither dataset currently supports a clean structured comfort-care sensitivity |
| Withdrawal/withholding proxy | Not documented as available | No approved structured candidate counted | Direct withdrawal/withholding sensitivity remains unsupported |
| Palliative-care proxy | Not documented as available | Present; common; descriptive/supporting only | MIMIC can describe palliative context, not treatment limitation itself |
| Hospice/discharge proxy | Not part of primary ASIC sensitivity | Present via `admissions.discharge_location`; weak discharge context | Discharge context should remain separate from ICU treatment limitation |
| Brain-death/organ-donation proxy | Not documented as available | Present as separate context domain | MIMIC can flag this pathway separately, not as treatment limitation |
| AMA/nonstandard discharge proxy | Flagging mentioned in cohort framing, not a direct treatment-limitation proxy | Checked; zero retained stays | Keep as discharge/process context only |
| Timestamp usability | Limited by ASIC absence of direct proxy fields | ICU items timestamped; ICD untimed; discharge fields post-event | MIMIC timing is partially usable but horizon alignment is still deferred |
| Supports hard-case sensitivity? | Not directly, unless explicit absence is carried as limitation | Weakly testable through code-status proxy sensitivity | Sensitivity support is asymmetric |
| Main limitation | Treatment-limitation/end-of-life confounding unresolved | Code-status signal is substantial but semantically incomplete | Similarity across datasets would not eliminate end-of-life confounding |

## Consequence for Chapter 1 interpretation

If MIMIC hard-case results differ from ASIC, treatment-limitation documentation differences are a plausible interpretation threat. If MIMIC hard-case results resemble ASIC, the MIMIC code-status sensitivity can partially strengthen the bounded interpretation, but it cannot eliminate end-of-life confounding. ASIC remains more limited for direct treatment-limitation sensitivity.

The primary Chapter 1 cohort and risk models should not be changed on the basis of these proxies. Proxy-positive stays should not be excluded from the primary analysis. These contrasts are interpretive and do not support causal claims.

## Recommended wording

Treatment-limitation and end-of-life processes were assessed in MIMIC-IV using available structured proxies for documented code-status limitation, palliative-care involvement, hospice/discharge context, brain-death/organ-donation pathways, and AMA/nonstandard discharge. These variables should be interpreted only as documented structured proxies: absence of a structured marker does not imply absence of treatment limitation.

In MIMIC-IV, documented code-status limitation was sufficiently common to support a bounded proxy sensitivity, but comfort-care and withdrawal/withholding processes were not captured as approved structured proxies. ASIC offered less direct structured support for this sensitivity, so treatment-limitation confounding remains an asymmetric and explicit limitation of the ASIC-MIMIC comparison.
