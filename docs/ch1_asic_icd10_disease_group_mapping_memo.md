# ASIC ICD-10 Disease-Group Mapping Inspection Memo

## Scope

This memo covers only the first ICD-10 mapping subtask for Phase 1 / Chapter 1 / Sprint 3 / Issue 3.2:

- inspect the raw ASIC static `icd10_codes` field
- propose a rule-based mapping basis for the frozen six-group hierarchy
- flag ambiguity points before any implementation

It does **not** implement the final stay-level disease-group variable, does **not** integrate anything into the hard-case comparison table, and does **not** broaden the frozen group scheme.

## Data Basis

- Inspected local standardized ASIC artifact: `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv`
- The full HPC path named in the issue (`/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full/static/harmonized.csv`) is not mounted in this workspace, so this memo is based on the locally available harmonized ASIC artifact only.
- Local inspected slice: `80` rows
- `icd10_codes` nonmissing: `80/80`

Implication: the field-format findings look stable enough to design the parser and rule logic, but rare prefixes present only in the full cluster artifact could still appear later. That is a residual QC risk, not a reason to expand the scheme now.

## Observed `icd10_codes` Format

### What the field looks like in practice

- In the inspected artifact, `icd10_codes` is always a **list of codes**, not a single principal diagnosis.
- Every nonmissing row is a **comma-separated string**.
- The field is often dense:
  - minimum `3` codes per row
  - median `22.5` codes per row
  - maximum `144` codes per row
- Duplicate codes within the same row are common: `33/80` rows.
- Raw order does **not** look reliable enough to encode diagnosis priority.

### Delimiters and formatting variation

- Main delimiter: comma
- Semicolons observed: `0`
- Pipe-delimited rows observed: `0`
- Rows with a space after the comma: `20/80`
- Rows without a space after the comma: `60/80`
- Lowercase observed: `0`
- Empty-string rows observed: `0`

### Token shape

- Almost all tokens are plain **3-character ICD-10 stems**.
- Observed token counts:
  - `1933` plain 3-character tokens
  - `16` decimal-form tokens
  - one nonstandard token form: `U07.1!`
- Decimal examples:
  - `J80.01`
  - `J80.02`
  - `J80.03`
  - `U07.1`
  - `U07.1!`
  - `U99.0`

### Practical granularity implications

- **3-character prefixes are practical** and dominate the field.
- Chapter-level matching is practical for some groups:
  - `J*` respiratory
  - `S/T/V/W/X/Y*` trauma or procedural complications
  - `A*` and `B*` infection
  - `G*` neurologic
- Chapter-level matching is **not** sufficient on its own for all groups:
  - `I*` mixes cardiovascular and cerebrovascular disease
  - postop complication families such as `J95` or `I97` need explicit handling

## Representative Raw Examples

- `I71,I72,K55,J18`
- `D41,I26,I48,J12,J80.03,J96,K80,U07.1,U99,Z11`
- `T81,R00,R77,U69,Z29,L89,E11,Z43,B37,Z93,N17,E03,E78,I10,E05,E23,J80.03,A49,Z11,I31,J96,B96,D68,D62,U07.1!,J12,D69,Z22`
- `S12,S22,S22,S24,S27,S27,S36,S36,S22,S22,S52,S02,F20,E10,S29,S12,S22,S52,T81,U50,U51,Z29,Z23,R57,D62,D69,D65,E87,F32,R63,E87,Z43,Z93,A41,A41,N39,J96,Z11,G96,T81,S24,U69,B95,R65,B96,U99`
- `I21, I25, R57, I50, I10, J96, E66, J18, U69, D68, N17, Z43, G72, E87, Z11, U99, D62, I49, G62, T81, L40, J90, I95, A41`

These examples show the core challenge: the field is a mixed diagnosis bag, not a clean single-disease field.

## Recommended Parsing Approach

Use a normalization-first parser before any disease-group logic:

1. Split on comma only.
2. Strip surrounding whitespace from each token.
3. Uppercase the token.
4. Remove internal spaces.
5. Strip a trailing exclamation mark if present, so `U07.1!` normalizes cleanly.
6. Derive:
   - chapter letter: first character
   - 3-character stem: first letter plus next two digits, ignoring decimals
7. Deduplicate within row before matching.
8. Match on the normalized **set** of codes, not on raw order.

Practical examples:

- `J80.03` -> chapter `J`, stem `J80`
- `U07.1!` -> chapter `U`, stem `U07`
- `U99.0` -> chapter `U`, stem `U99`

## Proposed Mapping Table

Frozen first-match hierarchy for later implementation:

1. surgical / postoperative / trauma-related
2. respiratory / pulmonary
3. infection / sepsis non-pulmonary
4. cardiovascular
5. neurologic
6. other / mixed / uncategorized

| target group | proposed ICD-10 prefixes / patterns | rationale | likely ambiguity | notes |
| --- | --- | --- | --- | --- |
| surgical / postoperative / trauma-related | any `S*`, `T*`, `V*`, `W*`, `X*`, `Y*`; plus postop complication stems `I97`, `J95`, `K91`, `M96`, `N99` | Best simple way to capture trauma, injury, poisoning, external-cause trauma coding, and explicit postprocedural complications seen in the sample (`T81`, `T84`, `T86`, `T89`, `S02`, `S06`, `J95`, `K91`, `M96`, `Y84`) | Many rows also contain `J*`, `A/B*`, `R65`, `I*`, or `G*`; the hierarchy intentionally lets this group win in mixed trauma/postop cases | Do **not** use generic `Z*` status or device codes such as `Z43`, `Z95`, `Z99` as surgical drivers |
| respiratory / pulmonary | any `J*` after the surgical rule has already fired; normalize `J80.01/.02/.03` to `J80` | Observed pulmonary burden sits overwhelmingly in chapter `J` (`J12`, `J13`, `J15`, `J17`, `J18`, `J44`, `J45`, `J69`, `J80`, `J90`, `J91`, `J93`, `J94`, `J96`, `J98`) | Pulmonary infection rows also contain infection codes; respiratory should win by design because the frozen infection group is explicitly non-pulmonary | Reserve `J95` for the earlier surgical/postop rule rather than treating it as ordinary respiratory disease |
| infection / sepsis non-pulmonary | `A*`, `B*`, `R65`, `U07` | Simple coarse capture of systemic and non-pulmonary infection burden; observed sepsis/infection markers include `A41`, `A49`, `B37`, `B44`, `B95`, `B96`, `B97`, `R65`, `U07` | This bucket will be smaller than raw infection prevalence because pulmonary infection is intentionally claimed earlier by `J*`, and postop/trauma infection is intentionally claimed earlier by the surgical rule | Exclude `N39` as a standalone infection trigger because 3-character `N39` is too broad once decimals are lost; also do not let `U80`, `U81`, `Z22`, or `Z29` drive infection on their own |
| cardiovascular | any `I*` **except** `I60-I69` and `I97`; plus `R57` | Captures the main observed cardiovascular burden (`I10`, `I21`, `I25`, `I31`, `I33`, `I35`, `I46`, `I48`, `I50`, `I71`) while keeping stroke out of the cardio bucket | `R57` may reflect septic shock; the earlier infection rule is what prevents obvious septic-shock rows from defaulting to cardiovascular | `I26` pulmonary embolism and related vascular codes remain here because there is no separate thromboembolic group in the frozen scheme |
| neurologic | any `G*`; any `I60-I69`; plus clear CNS neoplasm stems `C70-C72`, `D32-D33`, `D42-D43` | Keeps stroke and core neurologic disease together and catches clear intracranial neoplasm cases that would otherwise fall through | Trauma-associated neuro codes often co-occur with `S/T*` and should be claimed earlier by the surgical rule | Do **not** use symptom-only `R29`, `R40`, `R47` as standalone neurologic drivers |
| other / mixed / uncategorized | default fallback | Necessary because many rows are dominated by metabolic, renal, GI, oncologic, status, or nonspecific support codes without a clear frozen-group driver | This bucket will still contain clinically heterogeneous stays | Common non-driver codes in the sample include `Z11`, `Z43`, `Z95`, `Z99`, `U69`, `U80`, `U81`, `U99`, `E87`, `D62` |

## Ambiguity Notes

- **Multi-system lists are the norm, not the exception.** In the inspected local artifact, many rows hit three to five candidate groups at once once obvious patterns are checked.
- **The hierarchy is doing real work.** A large share of rows combine trauma or postop codes with respiratory failure, sepsis, cardiovascular disease, or neurologic disease. Without the frozen first-match hierarchy, the variable will not be reproducible.
- **Pulmonary infection vs non-pulmonary infection is the key intentional split.** Rows with `J*` plus `A/B*` or `R65` should remain respiratory, not infection. That is the only way to keep the infection bucket aligned with the frozen label `infection / sepsis non-pulmonary`.
- **Postprocedural complication codes collide with organ-system chapters.** `J95`, `I97`, `K91`, `M96`, and `N99` should be treated as surgical/postoperative, not as ordinary respiratory, cardiovascular, digestive, musculoskeletal, or renal disease.
- **`R57` is not clean.** At 3-character level, shock is too nonspecific to separate cardiogenic from septic shock. The hierarchy has to carry the clinical simplification here.
- **`N39` is too blunt at 3-character level.** Once decimals are lost, `N39` no longer cleanly means urinary infection, so it should not be used as a standalone infection trigger.
- **Z and U codes are common but mostly poor disease-group drivers.** Examples with high frequency in the sample include `Z11`, `Z43`, `Z95`, `U69`, `U99`, `U80`, `U81`, and `U99.0`. These look more like status, screening, resistance, epidemiologic, or auxiliary codes than primary disease-group anchors.
- **Decimals exist but do not justify a decimal-level scheme.** They are rare and can be collapsed to 3-character stems without losing the mapping logic.

## Recommended Coding Granularity

The simplest defensible later implementation is a **mixture**, centered on normalized 3-character stems:

- use **chapter-level rules** where the chapter is already coherent for the target group:
  - `S/T/V/W/X/Y*` for surgical / trauma-related
  - `J*` for respiratory
  - `A*` and `B*` for infection
  - `G*` for neurologic
- use **3-character exclusions or special families** where chapter-level matching is too blunt:
  - `I60-I69` vs other `I*`
  - `I97`, `J95`, `K91`, `M96`, `N99`
  - `R65`
  - `R57`
  - `C70-C72`, `D32-D33`, `D42-D43`

Recommendation: do **not** build the final mapping around decimal-level codes. They are too sparse in the inspected artifact to justify the extra complexity.

## Final Recommendation

The frozen six-group mapping looks **feasible** for ASIC as a coarse descriptive variable, but only if later implementation treats `icd10_codes` as a normalized multi-code set rather than a principal diagnosis field.

Recommended later implementation strategy:

- normalize to uppercase deduplicated tokens
- derive a 3-character stem for matching
- apply the frozen first-match hierarchy exactly as specified
- rely on a small mixture of chapter-level rules plus a short exception list
- avoid letting common `Z*` and auxiliary `U*` codes drive the grouping

Bottom line: freeze the mapping logic around **normalized 3-character stems plus a few chapter-level and postop/stroke exceptions**, not around raw string order and not around decimal-level ICD detail.
