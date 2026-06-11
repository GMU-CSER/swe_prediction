# Summary: why the retrained models fail on the 2025 test periods — a data pipeline issue, not a model issue

We reproduced the problem you described and traced it to its root cause. The short
version: the model is fine; the test-time inputs are far outside the training
distribution because of two corrupted feature groups, one on each side of the pipeline.

**1. The symptom reproduces with a known-good model.**
We took a GCN trained on the updated training graph with pure RMSE loss
(validation/test R² = 0.971 on the random split) and ran it on
`gnn_testing_data_2025-01-15.pt`. Predictions collapse exactly as you observed:
median 0.19 inches, maximum 3.0 inches across the whole domain. Since the same
weights score 0.94+ on the training graph, the failure must come from the test
inputs, not from training.

**2. Root cause A: the Slope column in the training CSV is degenerate.**
In the training CSV, Slope is essentially a constant: median 89.975, max 89.998 —
every point sits on a ~90 degree slope, which is physically impossible (and means
the model never learned anything from this feature). In the test CSVs, Slope
contains real terrain values (median 0.76, range 0.04–89.9). Because the
StandardScaler was fit on the training data (mean ≈ 90, std ≈ 0.29), the real test
slopes map to about **−309 standard deviations**, which saturates the first layer
and crushes the output toward a constant. This single feature is the dominant cause.

**3. Root cause B: the lagged fsca columns in the test CSVs contain MODIS flag values.**
fsca is a percentage (0–100), and the current-day fsca is clean on both sides
(max 93–97). But in the test CSV for 2025-01-15, `fsca_7` has **median = max = 250**,
which is the MODIS snow product cloud flag (codes ≥ 200 are flags, not
measurements). After scaling, fsca_1..7 sit at +6 to +14 standard deviations. The
training CSV does not contain these codes (max 93), so this contamination is on the
test-generation side — presumably the cloud mask was not applied before merging the
lagged satellite fields.

**4. Causal confirmation.**
Neutralizing only the Slope column of the test graph (setting it to the training
mean) restores the predictions to plausible mid-January magnitudes: median 4.9,
p95 23.5, max 38.7 inches. One feature column accounts for the collapse; the fsca
flags add further distortion on top.

**5. Why validation never catches this.**
The validation split is a random node split of the same training graph, so it
shares the training distribution — including the broken Slope column. Any pipeline
inconsistency between training and test generation is invisible to it. This also
explains why changing seeds and switching among MSE/RMSE/SWE losses made no
difference: all of them consume the same out-of-distribution test inputs.

**Proposed fixes (we can run all of this and send results):**
- Drop Slope from the feature set in both pipelines and retrain (it carries no
  information in the current training data anyway). Longer term, regenerate real
  slope for the training points — the test side shows the terrain extraction can
  produce correct values.
- In the test preprocessing, treat fsca values > 100 as missing (MODIS flags) and
  fill them from the nearest valid lag, applying the same rule on the training side
  for symmetry.
- Add a sanity check to the graph-building script: after scaling, assert every
  feature's |mean| and max |z| stay within bounds (e.g. 5 and 20), and fail loudly
  listing the offending columns. Either of these two bugs would have been caught by
  this check.

**Evidence and reproduction:** diagnostic script
`swe_prediction/code/diagnose_test_graph.py` (read-only, run as
`python diagnose_test_graph.py --date 2025-01-15`), full output archived at
`sweep_results/diagnosis_2025-01-15.log` on Hopper
(`/projects/kzhou6/bcui2/research/geo_project/`). Nothing under
`/groups/ESS/whung/` was modified.

---

# Update (2026-06-11): fixed pipeline implemented and validated

We implemented the proposed fixes (`code/data_process_fixed.py`, slurm job
`code/run_fixed_pipeline.sbatch`, outputs under `data_fixed/`, Wei-Ting's files
untouched) and retrained. The sanity gate immediately caught a THIRD bug we had
missed: **about 12.6% of test rows have the entire lagged meteorology record
zero-filled** (temperature 0 K, humidity 0%, wind 0 — a failed join filled with
zeros). The current-day columns were accidentally protected by the existing
temp<250 filter, but the lagged columns went straight into the features. The fix
was generalized: every lagged variable is masked against a physical validity
range (temperature [200,340] K, humidity [0.5,100]%, fsca [0,100], etc.) and
filled from the nearest valid lag; 33.6M invalid lag values were repaired in the
training CSV and 1.7M in the 2025-01-15 test CSV.

**Validation against 181 SNOTEL station observations on 2025-01-15**
(obs from `testing_all_ready_2025-01-15_paireddata_nopred.csv_snodas_mask.csv`,
matched by grid cell):

| pipeline | RMSE (inches) | R2 | corr | pred median / max (obs: 7.1 / 45.0) |
|---|---|---|---|---|
| broken (original test graph) | 10.52 | 0.006 | 0.12 | 0.22 / 1.28 |
| fixed | **5.62** | **0.354** | **0.60** | **6.19** / 17.07 |

The random-split test R2 of the retrained 84-feature model is 0.9675 (vs 0.971
with 85 features) — dropping Slope costs nothing. The predicted median now
matches the observed median (6.2 vs 7.1 inches; the broken pipeline predicted
0.22), and the prediction map (`sweep_results/fixed_prediction_map_2025-01-15.png`)
shows a realistic spatial snow pattern instead of an all-dark field.

## Multi-date validation (all 13 weekly test dates, 2026-06-11)

The fixed pipeline was applied to every weekly test date (2025-01-01 …
2025-03-26, `--test_only` mode reusing the saved scaler). Findings:

- **The lag contamination is systemic, not specific to 2025-01-15**: every
  date required 1.2M–3.6M invalid lag values to be repaired (MODIS flags +
  zero-filled meteorology).
- **All 13 dates now pass the sanity gate** after two principled calibrations:
  (a) the max|z| cap is per-column, max(30, 1.2 x the training column's own
  max |z|) — needed because zero-inflated precipitation legitimately reaches
  ~48 sigma inside the training data itself; (b) a handful of values beyond
  the training envelope (1–10 cells per column, < 0.01%, record atmospheric
  river storms in Feb/Mar 2025 exceeding anything in the 2019–21 training
  window) are winsorized to the envelope with a warning, while systematic
  |mean z| violations still abort. Note these storm values are real weather,
  out of the training distribution — winsorizing keeps the network in its
  operating regime.
- **Predictions are seasonally coherent across the winter** (a strong
  qualitative check): predicted max grows from ~20 inches in early January to
  ~46 inches in late March, following snow accumulation; p95 ranges 4.6–13.4
  inches. Station-based R2 can currently be computed only for 2025-01-15
  because the paired-observation file exists only for that date; generating
  paireddata for the other dates (Wei-Ting's obs pipeline) would enable a
  season-long skill series.

Full log: `sweep_results/multidate_validation.log`.

**Remaining (genuine) model issue:** the deep-snow tail is compressed —
predicted p95/max 12.7/17.1 inches vs observed 22.1/45.0. This is consistent
with the training label distribution (p99 = 27 inches, so the model rarely saw
deep snow) plus RMSE regression toward the mean, and is the legitimate target
of the "continue refining the model" discussion: e.g., tail-aware loss or
log-transformed target, more training coverage of deep-snow cells, and a
temporal-holdout validation protocol so that test-period skill is measured
during training. Note also that even within the training data, SNODAS SWE and
SNOTEL swe_value correlate only at 0.33, so SNODAS-based map comparisons have a
low ceiling and station observations remain the meaningful benchmark.
