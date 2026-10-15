# ROBT613 Midterm — Calibration-free P300 on BrainInvaders2013a

Does a P300 detector trained on *other* people work on a new user without calibration,
and how much does it lose compared with a subject-specific detector?

## Topic registration (paste into the team's row)

| Column | Value |
|---|---|
| Dataset | BrainInvaders2013a |
| Project idea | Cross-subject generalization |
| Hypothesis (one sentence) | A shrinkage-LDA P300 detector trained only on other subjects (leave-one-subject-out) reaches above-chance ROC AUC on unseen Brain Invaders users but stays below a subject-specific detector tested on the same held-out run, even when both use the same number of training epochs (paired Wilcoxon, 17 subjects). |
| Ambition | Challenging |
| Notes | Subjects 8–24 (single-session). Acquisition paper: Vaineau et al., 2019, arXiv:1904.09111. |

Registration sheet at the time of writing (2026-10-15): Team 1 EEGBCI, Team 2 BNCI2014-001,
Team 3 BNCI2014-004, Team 4 BNCI2014-009, no ideas filled in yet. Nobody holds
BrainInvaders2013a, so this row cannot conflict under the "same dataset **and** same idea"
rule. Re-check the sheet right before you add the row.

## Why this topic

* **Challenging tier.** Cross-subject transfer is one of the rubric's named Challenging
  questions (10% of the grade).
* **Grounded significance.** The dataset was recorded to compare calibrated
  (Non-Adaptive) and calibration-less (Adaptive) P300 play (Vaineau et al., 2019;
  Barachant & Congedo, 2014). We ask the same question with in-scope linear methods.
* **Enough subjects for a paired test.** 17 single-session subjects (IDs 8–24). A Wilcoxon
  signed-rank test cannot reach p < .05 with five or fewer subjects.
* **Fits the Colab budget.** About 2.2 GB for subjects 8–24. P300 features with shrinkage
  LDA train in seconds. Subjects 1–7 are excluded because MOABB downloads all eight of
  their sessions (about 1 GB each).
* **Reliable signal.** The P300 is robust, so the neurophysiological check is not a gamble.
  The project's risk sits in the hypothesis, where it belongs.

## Experimental design (one factor at a time)

Every subject has two Training runs (Non-Adaptive and Adaptive), each with 80 target and
400 non-target flashes. Each run serves once as the **test run**:

| Condition | Training data | Test data |
|---|---|---|
| Within (subject-specific) | the subject's other run | held-out run |
| LOSO (calibration-free) | all runs of the other 16 subjects | same held-out run |
| LOSO-matched (control) | random subset of the other subjects' epochs, sized like Within | same held-out run |

The comparison also covers:

* **Same in every condition:** the preprocessing, the features (decimated ERP 0–0.8 s,
  16 channels), the pipeline (`StandardScaler` → shrinkage LDA) and the metric (ROC AUC).
* **Chance level:** label permutation per subject.
* **Statistics:** paired Wilcoxon across subjects, with effect size and per-subject values.
* **Secondary question:** do the per-subject P300 latency and amplitude (stage 4) explain
  who LOSO fails?

**Leakage rules.** No labels and no epochs of the test subject enter LOSO training. One
label-free exception: a channel flagged bad in *any* subject (`GLOBAL_BADS`, a channel-quality
flag) is dropped from every subject's features, so all subjects share one channel set. It
is empty for subject 8; the full run reports it. Filters and ICA are fitted per run, without labels, on that run's own data. Rejection thresholds are
fixed before decoding. Hyper-parameters are not tuned on test runs: shrinkage is set
analytically with `shrinkage='auto'`, so there is nothing to tune.

## What the audit on subject 8 already showed (local run, 2026-10-15)

* **Data source trap.** MOABB 1.7.2 downloads from the NEMAR mirror by default. For
  BI2013a, that mirror holds only `3.mat` per session (Non-Adaptive Training), so the
  Adaptive Training run (`1.mat`) would be missing and no error would be raised. The
  notebook sets `MOABB_DOWNLOAD_PROVIDER=upstream` to get the full Zenodo archives.
* **Runs.** There are two training runs, with run names coming from the file number.
  Run `0` is `1.mat`, Adaptive. Run `2` is `3.mat`, Non-Adaptive. The run order is
  randomised per session, so the notebook reads each subject's `meta.yml` to label the
  conditions.
* **Counts.** Each run has 80 target and 400 non-target flashes, with codes 33285/33286 and
  no other codes. The median flash interval is about 350–400 ms.
* **Units.** The data are already in volts. DC offsets reach about 20 mV because the
  amplifier is DC-coupled. The 1–20 Hz SD is about 10 µV on posterior channels and about
  40–50 µV on Fp1/Fp2 (blinks).
* **Noise.** 50 Hz mains noise is strong, worst on P7 at about +30 dB, and blinks are
  visible on the frontal channels. These feed stage 2 (notch or low-pass) and stage 3
  (ICA with Fp1/Fp2 as proxies).
* **Provenance conflict.** The MOABB docstring says TMSi Nexus. The paper and MOABB's
  `METADATA` say g.USBamp (g.tec) with ground at Fz. The notebook follows the paper.
* **Quick check, not part of the pipeline.** Target − non-target at Pz is about +8–11 µV.
  The P300 is there.

The Within condition needs exactly one Adaptive and one Non-Adaptive training run per
subject. The notebook asserts this, and the counts, for every audited subject.

## Stage 2 decisions (subject 8, local run; the full table comes from Colab)

| Choice | Value | Main evidence |
|---|---|---|
| Band-pass | FIR 0.5–20 Hz, Hamming, zero-phase, 3381 taps (6.6 s) | 73–80% of 0–20 Hz power is drift below 0.5 Hz. On the same flashes, the 0.5 Hz P300 peak keeps 90–102% of the 0.1 Hz peak (from 0 µV or from the pre-flash mean). 1 Hz keeps 76–94%; 2 Hz keeps 61–73% and distorts the shape. A 30 Hz low-pass changes the difference wave by only ~7% RMS. |
| Filter type | FIR, not IIR | Linear phase applied zero-phase keeps latencies; always stable (notebook 03). The length is handled by the edge guard. |
| Notch | none | 50 Hz on P7 drops from 61–83 µV RMS to 0.01–0.03 µV, ≤0.16% of in-band RMS (−75 dB at 50 Hz). |
| Resampling | 512 → 128 Hz, events resampled with the data | Max error 0.04 µV against ~9 µV in-band RMS. Counts stay 80/400. Timing changes by ≤3.9 ms. |
| Edge guard | drop flashes within ±3.3 s of a recording edge | Truncation test: error is above 1 µV only within ~2.8 s of an edge. Costs 3 of 960 flashes (a 0.1 Hz high-pass would cost 34). |

Each run is filtered on its own, so no filter support crosses into the held-out run. The
whole notebook (audit + stage 2) runs locally in about 9 s with about 0.6 GB of RAM for
one subject.

## Stage 3 decisions (subject 8, local run; the full QC table comes from Colab)

| Choice | Value | Main evidence |
|---|---|---|
| Ocular cleaning | ICA (FastICA, 15 components, notebook 04) fitted per run on a 1–20 Hz copy and applied to the 0.5–20 Hz data | Fp1 target − non-target difference drops from 22.8 to 4.6 µV. Pz P300 unchanged (8.16 µV). |
| Component rule | Fp1/Fp2 correlation outlier (z ≥ 3) **and** ≥50% of map power on Fp1/Fp2/AFz | Removes 2 components (run 0) and 1 (run 2). Keeps a frontal component that does not correlate with the proxies. |
| Bad channels | flat (<0.5 µV), or SD more than 3× or less than ⅓× the median of the 14 non-proxy channels → dropped from every subject's features (channel selection, in scope; no interpolation); more than 2 bad → manual review | None for subject 8 (largest ratio 1.30). A robust z rule over-flagged P7 at a 1.3× SD and was dropped. |
| Reference | keep the left earlobe | An average of 16 sparse channels, two of them blink-dominated, would spread artifacts (notebook 04). |
| Baseline | none | The previous flash is a target before ≈20% of non-targets but only 1–4% of targets, so a baseline would differ by class. |
| Rejection | peak-to-peak 100 µV after ICA | 4 of 957 epochs dropped, with equal class curves. Before ICA the same threshold would drop 227, with 36% of targets against 21% of non-targets. |

**Confound finding.** Subject 8 blinks more after targets: 18–33% of target epochs against
12–19% of non-target epochs. Without ICA a decoder could partly read the eyes. The notebook
runs a group Wilcoxon test on blink fractions once all 17 subjects are loaded.

## Stage 4 findings (subject 8, local run; the group test comes from Colab)

| Check | Result |
|---|---|
| Predeclared window | 250–600 ms at Pz (notebook 01; dataset paper: 240–600 ms), fixed before looking |
| Window-mean T − NT at Pz | +3.14 µV, label-permutation p = 0.0005 (2000 shuffles, minimum possible) |
| Shape | Two positive peaks (≈240 and ≈390 ms; max 8.15 µV at 391 ms) over Cz/Pz/P3/P4, centro-parietal topography, plus an earlier ≈170 ms negativity at Oz/P4 |
| Consistency | The ERP image at Pz shows the bands across many single target epochs |
| Saved for later | Per-subject peak amplitude, peak latency, and similarity to the other subjects' grand average. These will be used to explain who the LOSO decoder fails. |

## Stage 5 (subject 8, local run; all subjects on Colab)

| Choice | Value |
|---|---|
| Features | mean amplitude in 16 × 50 ms windows over 0–800 ms (notebooks 01–02 window means, finer), 14 channels (Fp1/Fp2 left out as ocular proxies) = 224 features |
| Pipeline | `FunctionTransformer` → `StandardScaler` → `LDA(solver='lsqr', shrinkage='auto')`, all inside one `Pipeline` |
| Tuning | none: Ledoit–Wolf shrinkage is computed analytically inside each training fold |
| Within-subject CV | `LeaveOneGroupOut`, group = run |
| Chance | `permutation_test_score`, 200 shuffles within run |

**Result, subjects 8–10.** Within-subject AUC is 0.930, 0.927 and 0.944 (0.934 ± 0.009),
each with permutation p = 0.005. Subject 9's F6 is bad, so the features are 13 channels ×
16 windows = 208. The class information sits at 200–450 ms over Cz/P3/Pz/P4, with nothing
before 150 ms.

## Stage 6 smoke test (subjects 8–10, so LOSO trains on only 2 people; not the result)

| Subject | Within | LOSO-matched | LOSO |
|---|---|---|---|
| 8 | 0.930 | 0.740 | 0.779 |
| 9 | 0.927 | 0.750 | 0.784 |
| 10 | 0.944 | 0.743 | 0.806 |

* **Direction of each hypothesis.**
  * H1: LOSO beats chance on all 6 test runs (perm p = 0.001).
  * H2: Within > LOSO, with a median gap of 0.143 against a fold SD of 0.014.
  * H3: Within > LOSO-matched (0.93 vs 0.74 with equal data), so the gap comes from *whose*
    data, not *how much*.
* **Inference.** None yet. With n = 3 the Wilcoxon minimum is p = 0.125. The full Colab run
  (LOSO on 16 people) decides.
* **Runtime.** Stage 6 for 3 subjects takes 0.6 s. The whole notebook for 3 subjects takes
  ≈30 s locally.

**Carry forward into later stages**
* **LOSO-matched control.** Match the Within epoch count *after* the edge guard. Run 2 of
  subject 8 already loses 1 target and 2 non-target flashes.
* **Event timing.** Events move by up to 3.9 ms when resampled to 128 Hz. If that ever
  matters, epoch at 512 Hz with `decim=4` instead, which keeps exact timing.
* **Stage 6.** Reuse `PIPE`, `to_xy` and `within_folds` (AUC per subject and test run).
  LOSO trains on all runs of the other 16 subjects and tests on the same held-out run.
* **Stage 6 chance level without refitting.** The LOSO model never sees the test subject.
  So shuffling only the test-run labels against its fixed scores gives a valid AUC null
  at almost no cost. Do not copy `permutation_test_score` into LOSO: 200 refits on ~15k
  epochs × 34 folds would break the 30-minute budget.

## Roles (from the project page)

| Role | Leads | Reviews | Stages |
|---|---|---|---|
| Data and provenance | loading, audit, events, subject selection | Evaluation | 1 |
| Signal processing | filtering, ICA (Fp1/Fp2 as EOG proxy), QC | Data | 2, 3 |
| Decoding | ERP features, pipeline, Within/LOSO splits | Signal processing | 4, 5 |
| Evaluation and reporting | permutation chance, Wilcoxon, figures, report | Decoding | 6, 7 |

## Plan (3 weeks)

**Week 1**
* Register the topic.
* Run the full audit on Colab (all 17 subjects) and record the download and load time.
* Stage 2: PSD and filter choice.
* Read 2–3 papers for the significance section.

**Week 2**
* Stage 3: ICA and rejection counts.
* Stage 4: ERP check and per-subject P300 table.
* Stage 5: the pipeline, sanity-checked on one subject.

**Week 3**
* Stage 6: the three conditions, permutation tests and the Wilcoxon test.
* Fresh-Colab timing run.
* Write the report (IEEE, 6–8 pages) and the 12-minute slides.

## Full run on Colab (stages 1–6, all 17 subjects)

1. Upload `notebooks/project.ipynb` to Colab with a CPU runtime and choose
   **Runtime → Run all**. The first cell installs mne 1.13.2 / moabb 1.7.2. If pip asks for
   a restart, restart and run all again.
2. Expect a download of about 2.2 GB from Zenodo. The audit table's `load_s` column records
   how long it took.
3. The `assert`s are deliberate. If one fails, keep it and send the message back. They
   check:
   * two training runs of 80/400 per subject;
   * `events_kept`;
   * `review`;
   * that no core P300 channel (Cz/Pz/P3/P4/Oz) is in `GLOBAL_BADS`;
   * that `GLOBAL_BADS` has at most 2 channels.
4. In the outputs, look at:
   * runs where `ica_excluded` is `-` or lists 3 or more components;
   * any non-empty `bad_channels` and `GLOBAL_BADS`;
   * the group blink Wilcoxon in 3.2 and the group ERP Wilcoxon in stage 4;
   * the retained percentage;
   * the stage 6 table of H1–H3, and the Spearman correlations in 6.2;
   * **Total runtime** at the end of the notebook (stage 7) against the 30-minute budget.
5. Copy the package versions printed at the end into the report. The scikit-learn version
   matters for the shrinkage LDA.
6. All group-level branches (n ≥ 6) were run locally once on 3 subjects with
   `BCI_MIN_GROUP_N=2` and finished without errors. Their p-values there mean nothing.

## Layout

```
notebook_src/project.py    # source of truth (percent format, diff-friendly)
notebooks/project.ipynb    # built notebook — submit with outputs
tools/build_notebook.py    # project.py → project.ipynb
requirements.txt           # pinned to the workshop versions
```

Build the notebook after editing the source:

```bash
python tools/build_notebook.py notebook_src/project.py notebooks/project.ipynb
```
