# Speaker notes

## 1. Can a P300 decoder skip calibration?

45 s. Introduce the practical question: can a new user avoid labelled calibration? Make clear that this is an offline, retrospective analysis of BrainInvaders2013a, not a new online BCI. Introduce the four team members and their lead areas. Source: notebook introduction and results/run_manifest.json.

## 2. Can other users’ data replace calibration?

60 s. Every P300 session normally starts with calibration. H1 asks whether training on other people beats chance. H2 compares transfer with subject-specific calibration. H3 matches training class counts so that quantity alone cannot explain a gap. Barachant and Congedo (2014) motivate generic initialization; Kim and Kim (2025) motivate practical zero-training use with methods outside our scope, which we do not implement. Sources: arXiv:1409.0107 and doi:10.1109/TNSRE.2025.3603979.

## 3. Data audit and provenance

60 s. BrainInvaders2013a, subjects 8–24: each has exactly one session. Subjects 1–7 have eight sessions each (about 1 GB per subject), which does not fit the 30-minute Colab budget, so they are excluded by data size, not by performance. Every included run passes assertions for 80 targets, 400 non-targets, 16 EEG channels and 512 Hz. Units are volts in MNE, reported in µV. Data come from the original Zenodo files and are loaded through MOABB into MNE. Source: notebook stage 1; acquisition paper arXiv:1904.09111.

## 4. Filter choice backed by the spectrum

65 s. The PSD shows slow drift below 0.5 Hz and a sharp 50 Hz mains peak, strongest on P7. The FIR band-pass 0.5–20 Hz removes both: after the low-pass, 50 Hz is about −75 dB, so a notch adds nothing (measured in filter_measurements.csv). Data are resampled from 512 to 128 Hz together with the events. A 3.3 s edge guard (the filter half-support) keeps epochs away from recording edges. Filtering is done per run, so nothing crosses held-out runs. Zero-phase filtering uses future samples, so the analysis is offline. Source: notebook stage 2.

## 5. Artifact handling and quality control

65 s. ICA (FastICA, 15 components) is fitted on a 1–20 Hz copy of each unlabelled run and the unmixing is applied to the main band-pass. Fp1/Fp2 serve as ocular proxies because there is no EOG channel. A component is removed only if both a proxy-correlation outlier rule and a frontal-map rule agree. The plot shows the blink-related deflection at Fp1 disappearing while the Pz response is preserved. Then a 100 µV peak-to-peak rejection runs on the cleaned epochs. Full-cohort counts: 3 target / 12 non-target edge drops and 80 / 329 amplitude drops; 97.40% of epochs retained. F6 was flagged as a bad channel and is ignored by rejection; it is not in the decoder montage. Residual artifacts remain possible. Source: notebook stage 3 and qc.csv.

## 6. The P300 is there before we decode it

65 s. Before any classifier, check that the expected brain response exists. The Pz window 250–600 ms was fixed in advance (notebook 01; Vaineau et al. 2019 report 240–600 ms). Grey lines are individual subjects, red is the grand average; the topomaps show the centro-parietal positivity peaking around 350 ms. 16 of 17 subjects have a positive window mean, mean contrast 2.74 µV. Single-epoch permutations are descriptive only because flashes overlap. The window is never chosen by decoding accuracy. Source: notebook stage 4 and erp.csv.

## 7. One leakage-free pipeline

55 s. A feature is the mean voltage at one electrode during one 50 ms interval: five sites times sixteen windows over 0–800 ms gives 80 features. The montage was frozen before the full run and does not depend on held-out data. Everything that learns from data — the scaler and the analytic shrinkage of LDA — sits inside one scikit-learn Pipeline and is fitted on training folds only. No hyperparameters are tuned, so no nested CV is needed. The heatmap shows where the target − non-target difference lives for subject 8. Source: notebook stage 5.

## 8. Same test epochs, different training data

65 s. All three conditions test exactly the same held-out run of each subject. Within trains on the subject’s other run (leave-one-run-out, never a random epoch split, because overlapping flashes share samples). LOSO trains on all runs of the other 16 users, and an assertion guarantees the test subject is never in training. LOSO-matched subsamples the other users to the same target and non-target counts as Within and averages ten seeded draws. Chance level comes from 200 label permutations within runs. Each subject’s score is the mean over its two test runs. Source: notebook stages 5–6 and comparison_runs.csv.

## 9. Transfer works, calibration still wins

65 s. Bars are means over 17 subjects; each subject is the mean of its two held-out runs, so the SD is between-subject spread. Within reaches 0.875, LOSO 0.810, LOSO-matched 0.724. LOSO-matched is lower than full LOSO, so the amount of external data matters. All 34 LOSO test runs beat their permutation null at p < .05. No bar comes from a random epoch split. Source: comparison_subjects.csv.

## 10. Every user transfers, but not equally

65 s. Each grey line is one person (mean of two test runs); the coral line is the cohort mean and the slate line is chance. 15 of 17 users lose AUC without calibration. The median Within − LOSO gap (0.072) is several times the median Within fold SD (0.018), though two folds estimate variability poorly. The weakest LOSO subject still reaches 0.671. Source: comparison_subjects.csv.

## 11. All three hypotheses hold in this cohort

65 s. One subject is one unit. The three directional comparisons use paired Wilcoxon signed-rank tests with Holm correction. Read the median difference together with the adjusted p-value. Excluding pilot subjects 8–10 (n = 14) gives the same conclusions. These are conventional cohort summaries: LOSO models share training users, so p-values are not exact guarantees for new populations. Source: hypothesis_tests.csv and sensitivity_tests.csv.

## 12. What the result shows

45 s. State the outcome as conditional cohort evidence: transfer works well above chance, but calibration still adds about 0.07 AUC, and the amount of external data matters. Then stress the limits: offline preprocessing, flash-level AUC instead of command accuracy, one acquisition protocol, and dependent LOSO models. A negative result would have been equally valid under the rubric. Every teammate should be ready to explain any stage. Source: notebook conclusions and run_manifest.json.

## 13. References and AI disclosure

Backup slide. References are background and motivation, not sources of results. The core pipeline follows workshop notebooks 00–06 and cites them where code is reused. AI tools: OpenAI Codex for code review, debugging, API lookup, implementation, statistics checks, plotting and text drafting; Claude Code for checking the submission against the assignment and editing wording. Every number comes from the executed notebook.

## 14. Reproducibility record

Backup slide. Direct dependencies are pinned in requirements.txt and actual versions are recorded in the run manifest. The verified run used a fresh Google Colab CPU session with an empty data cache: 7.41 minutes including downloads, all 32 code cells, no errors. The first install needs one runtime restart to reload NumPy. The downloader respects Zenodo rate limits. Source: results/run_manifest.json and results/colab_verification.json.
