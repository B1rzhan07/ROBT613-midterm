# Calibration-free P300 on BrainInvaders2013a

ROBT613 Midterm Project. Birzhan Zhunusbekov, Arman Abakov, Nazerke Khamzina,
Bekmurat Amangeldiyev. Topic registered 2026-10-15: BrainInvaders2013a,
cross-subject generalization, Challenging.

## Study

Does a shrinkage-LDA P300 decoder trained on other people work above chance on a new user,
and how much does it lose relative to subject-specific training on the same test epochs?
The matched control holds target and non-target training counts constant.

Subjects 8-24 each contribute two training runs. Features are 16 window means over
0-800 ms at fixed Cz/P3/Pz/P4/Oz sites. The same scaler and analytic-shrinkage LDA pipeline
runs in all conditions. Within splits by run; LOSO excludes the entire test subject.
Three directional subject-level hypotheses use Holm-adjusted Wilcoxon tests.

Subjects 8-10 informed development. The full study therefore includes a sensitivity summary
excluding those subjects from evaluation. This is a retrospective, pilot-informed study.
Unlabelled ICA fits each whole run and filtering is zero-phase: results are offline.
Epoch-level permutation p-values are descriptive because neighbouring flash responses overlap.

## Reproduce locally

From this folder, with Python 3.12 and an isolated virtual environment:

```bash
python -m pip install -r requirements.txt nbformat nbclient ipykernel
python -m ipykernel install --user --name bci-midterm --display-name 'BCI midterm (pinned)'
python tools/run_notebook.py
```

The runner builds the self-contained notebook from `notebook_src/project.py`, executes it,
saves outputs, exports plot images, and writes tables plus a run manifest under `results/`.
It saves the partial notebook if execution fails. Assertion failures require investigation;
do not remove checks just to force completion. Re-running `tools/build_notebook.py` alone
clears notebook outputs and must be followed by execution before submission.

Only unchanged training MAT members and metadata are fetched from original Zenodo ZIPs.
ZIP CRC verifies extraction and SHA-256 manifests record downloaded bytes. MOABB still
loads every EEG file into MNE. Cached original MOABB files are reused. A fresh notebook
contains this helper, so Colab does not require copying the tools directory.

## Deliverables and status

See `SUBMISSION_CHECKLIST.md` for final file paths. `CONTRIBUTIONS.md` is the dated
contribution log, together with the Git history.

The report uses IEEEtran. Rebuild it after notebook execution with:

```bash
python tools/build_report.py
tectonic report/main.tex
```

The report and slides use exported notebook numbers, not hand-entered estimates. Direct
analysis dependencies are exactly pinned in `requirements.txt`. `results/run_manifest.json`
records actual versions, subjects and execution environment. `requirements-lock.txt`
records the broader local execution environment when supplied.

The [saved, executed Colab notebook](https://colab.research.google.com/drive/19qYoD5urhaAcdMhhokzYAZPZuN2ra2Rh) passed on a restarted free CPU session with an empty data cache: **7.41 minutes** including setup checks, downloads and analysis, all 17 subjects, no errors. Pinned packages were installed before the initial restart. See `results/colab_verification.json` and `REPRODUCIBILITY_NOTES.md`.

The GitHub launch URL in the checklist opens the notebook published in this repository.

## Academic integrity

The notebook cites each reused ROBT613 workshop stage (B. Abibullaev,
https://github.com/berdakh/ROBT613-BCI-Workshop). AI tools are disclosed in the last
notebook section and in the report appendix: OpenAI Codex (code review, debugging,
implementation, plotting, text drafting) and Claude Code (requirements check, wording
edits). Every number comes from the executed notebook. The original dataset papers
motivate the question; no out-of-scope Riemannian or neural-network method enters the
core pipeline.
