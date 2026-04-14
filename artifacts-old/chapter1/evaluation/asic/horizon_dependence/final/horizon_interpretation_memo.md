# Horizon Interpretation Memo

This memo is intentionally bounded to the frozen Chapter 1 ASIC logistic last-eligible stay-level design.
All numeric values below come from the repository's synthetic stand-in data and are implementation-test outputs only.

## Readout

- Hard-case share: `8h` to `24h` stays at `0.40` to `0.40`, then is higher at `48h` `0.50` and `72h` `0.60`. For the narrative anchor and main contrast, `24h` is `0.40` and `48h` is `0.50`.
- Hard-case membership: overlap is substantial but incomplete. Mean pairwise Jaccard is `0.690`; `24h` vs `48h` has Jaccard `0.500`, with directional overlap `24h -> 48h` `0.750` and the reverse `0.600`.
- Mortality-vs-risk shape: the five binned panels still show mortality increasing with risk, but the pooled-bin profile shifts enough between 24h and 48h to count as a material descriptive change in this local run. The weighted 24h vs 48h shape distance is `0.256`.
- Overall label: `change form`. On the local synthetic run, the balance between share, membership overlap, and the binned mortality-vs-risk panels is not cleanly captured by a simple persistence story, so change form is the closest label.

## Answers To The Four Questions

1. Hard-case share stable across horizons? `roughly yes, but modestly higher at longer horizons`.
2. Hard-case membership stable across horizons? `substantial but not perfect overlap`.
3. Mortality-vs-risk shape shift materially? `yes, enough to look materially different`.
4. Do hard cases persist, shrink, or change form? `change form`.

## Caveat

- The plotted horizon-level counts, thresholds, and overlap denominators were consistent with the saved Package 1 and Package 2 artifacts.
- Any chapter interpretation on real data must remain conditional on the saved horizon-specific q75 thresholds, the last-eligible stay-level representation, and the exact matched-denominator logic from Package 2.
- Local synthetic values in this memo are not scientifically interpretable.
