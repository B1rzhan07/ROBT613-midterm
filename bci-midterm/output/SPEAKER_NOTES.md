## 1. Calibration-free P300 decoding

45 s. Introduce the practical question: can a new user avoid labelled calibration? Explain that this is an offline retrospective analysis of BrainInvaders2013a, not a new online BCI. Names supplied by the team. Source: notebook introduction and results/run_manifest.json.

## 2. Research question and hypotheses

60 s. H1 asks whether external-subject training beats chance. H2 compares transfer with subject-specific calibration. H3 matches training class counts so quantity alone cannot explain a gap. The 2014 information-geometry paper motivates generic initialization and adaptation. Kim and Kim 2025 motivates practical zero-training use, using out-of-scope methods that we do not implement. Sources: https://arxiv.org/abs/1409.0107 and https://doi.org/10.1109/TNSRE.2025.3603979 (abstract consulted).

## 3. Data and provenance

60 s. Subjects 8-24 each have one session. Explain why subjects 1-7 are excluded: their longitudinal archives are substantially larger. Every included run must pass assertions for 80 targets, 400 non-targets, 16 EEG channels and 512 Hz. File metadata determines adaptive condition. All data still load through MOABB into MNE. Source: notebook stage 1; acquisition paper https://arxiv.org/abs/1904.09111.

## 4. Filtering and resampling

65 s. Explain evidence for 0.5-20 Hz rather than presenting defaults. Slow drift motivates high-pass filtering. The low-pass suppresses mains activity, assessed in residual measurements. Events resample together with the data. A half-support edge guard keeps epochs away from recording boundaries. No filtering crosses held-out runs. Zero phase uses future samples, so the conclusion is offline. Source: notebook stage 2 and filter_measurements.csv.

## 5. Artifact handling and quality control

65 s. ICA fits a 1-20 Hz copy of each unlabelled run and applies the unmixing to the main band-pass. Fp1/Fp2 are ocular proxies, not measured EOG. Exclusion requires both a proxy correlation outlier and a frontal map. The example plot is not a claim that every ocular artifact disappears. Present the full-cohort retention counts at right. A flagged core site or excessive bad channels stops the notebook for review. Source: notebook stage 3 and qc.csv.

## 6. P300 physiology before decoding

65 s. Establish a brain-signal check before classification. The preselected Pz window is 250-600 ms. Grand-average plots weight subjects equally. Point to the positive centro-parietal response and the individual variation. Single-epoch permutations are descriptive because flashes overlap. The physiological check never selects decoder features by their test accuracy. Source: notebook stage 4 and erp.csv.

## 7. Features and linear classifier

55 s. Describe a feature as the mean voltage at one electrode during one 50 ms interval. Five sites times sixteen windows gives eighty features. The montage was frozen before the full run and does not depend on held-out flags. The earlier three-person pilot informed the revision. Scaling and shrinkage fit training data only. No hyperparameters are tuned against test AUC. Source: notebook stage 5.

## 8. Evaluation on the same held-out epochs

65 s. All rows test exactly the same epochs for each subject/run. Within uses the other run from that person. LOSO uses all other people, with an assert excluding the test subject. Matched transfer preserves both target and non-target training counts and averages ten seeded draws. AUC ranks flashes and is distinct from complete command accuracy. Source: notebook stages 5-6 and comparison_runs.csv.

## 9. Mean AUC across subjects

65 s. State that the native chart uses subject means, and the printed SD is between-subject spread rather than fold spread. No bar represents a single random split. Interpret the observed ordering without assuming the hypothesis must succeed. Source: comparison_subjects.csv; exact values also appear in the report.

## 10. Individual transfer performance

65 s. Each thin line is one person, averaged across two test runs. The thick line is the cohort mean, and the dotted line is chance. Highlight variation rather than promising universal performance. Compare the median paired gap with the two-fold spread, while noting that two folds estimate variability poorly. Source: comparison_subjects.csv.

## 11. Paired hypothesis tests

65 s. One subject is one unit. The three directional primary comparisons use Wilcoxon signed-rank tests and Holm correction. Read both the median difference and adjusted p-value. Pilot-excluded sensitivity uses fourteen evaluation subjects. The tests are conventional cohort summaries: LOSO models share training users, and p-values are not exact guarantees for new populations. Source: hypothesis_tests.csv and sensitivity_tests.csv.

## 12. Conclusion and limits

45 s. State the outcome as conditional cohort evidence. Stress the limits of flash AUC, offline preprocessing and one acquisition protocol. A different outcome would still be valid under the project rubric. Explain the verified Colab CPU timing in the backup record. Every teammate should be prepared to explain all stages. Source: notebook conclusions and run_manifest.json.

## 13. References and AI disclosure

Backup slide. Acquisition and motivation references are background, not imported results. The core follows workshop notebooks 00-06. Codex assisted with implementation, execution, API lookup, statistics, plotting and drafting in this session. All numbers come from the executed notebook. Earlier AI usage remains for the team to confirm. Authors must review and be able to explain every line.

## 14. Reproducibility record

Backup slide. Exact direct dependencies are pinned and actual versions are in the manifest. The notebook saves figures and CSVs. Execution environment: Google Colab. Cold cache at setup: true. Initial installation required one restart to reload NumPy. The downloader respects Zenodo rate limits. Source: results/run_manifest.json. Saved notebook: https://colab.research.google.com/drive/19qYoD5urhaAcdMhhokzYAZPZuN2ra2Rh
