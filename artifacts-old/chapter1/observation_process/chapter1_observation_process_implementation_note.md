# Chapter 1 Observation-Process Variable Note

## Final Group Mapping Used
- `HR group`: `heart_rate`
- `BP group`: `map`, `sbp`, `dbp`
- `Respiratory group`: `resp_rate`
- `Oxygenation group`: `spo2`, `sao2`

## Derivation Rules
- Input source: raw `dynamic/harmonized` ASIC measurements plus unique usable 8-hour blocks.
- Block membership uses the existing upstream ASIC blocked-data contract: raw measurements with `minutes_since_admit` in `[block_start_h, block_end_h)`.
- `obs_*_grp_block` equals 1 when any raw group measurement is observed inside the current 8-hour block; otherwise 0.
- `n_core_grps_obs_block` is the row-wise sum of the four binary group indicators.
- `tsl_*` equals `prediction_time_h - latest_raw_observed_time_h` within stay and group, using all raw history with `minutes_since_admit < prediction_time_h * 60`.
- If a group has never been observed up to prediction time, the corresponding `tsl_*` stays missing.

## Deviations From Requested Design
- None in the derived variables themselves.
- The block-level export is unique per usable 8-hour block before horizon duplication; separate optional merged model-ready artifacts duplicate the block features across horizon-specific prediction-instance rows for convenience.

## Limitations / Ambiguities
- The raw-history derivation depends on `minutes_since_admit` in the harmonized ASIC dynamic table being aligned with the blocked 8-hour artifacts.
- Exact boundary handling was chosen to match the upstream blocked `*_obs_count` contract and verified empirically against the blocked feature table.