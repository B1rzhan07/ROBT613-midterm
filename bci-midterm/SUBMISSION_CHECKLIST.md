# Submission checklist

## Files

- Executed notebook: `notebooks/project.ipynb` (all 17 planned subjects, outputs saved).
- IEEE two-column report: `output/pdf/report.pdf`; editable source: `report/main.tex`.
- Presentation PDF: `output/pdf/presentation.pdf`.
- Editable presentation: `output/presentation.pptx`.
- Numerical evidence: `results/*.csv` and `results/run_manifest.json`.
- Contributions: `CONTRIBUTIONS.md`; Git repository: https://github.com/B1rzhan07/ROBT613-midterm.

## Human and cloud items

- [ ] Confirm each member's actual contributions and review assignments in CONTRIBUTIONS.md.
- [ ] Review all generated writing, citations, analysis choices and AI disclosure.
- [x] Fresh Colab CPU run saved: 17 subjects, 32 code cells, no errors, 7.41 minutes including setup and empty-cache data retrieval.
- [x] Saved Colab notebook URL recorded below; include it with the instructor submission.
- [ ] Confirm the team row in the registration sheet matches the supplied 2026-10-15 registration.
- [ ] Rehearse the 12-minute presentation with all four members.
- [ ] Each member submits their own confidential peer evaluation.
- [ ] Upload the required team files to the instructor's submission destination.

## Saved Colab notebook

https://colab.research.google.com/drive/19qYoD5urhaAcdMhhokzYAZPZuN2ra2Rh

Uploaded with user approval on 2026-10-27. Verified on a restarted free CPU session with an empty dataset cache: 7.41 minutes for setup plus analysis (7.27 minutes after initial setup). All 32 code cells executed without errors and saved 17 plots. The timed successful run used the already installed pinned packages after the initial NumPy restart. See results/colab_verification.json.

The saved notebook currently uses the owner’s existing access permissions. Confirm that the instructor can open it before submission. No sharing permissions were changed.

## Colab launch and verification

The repository launch URL is:
https://colab.research.google.com/github/B1rzhan07/ROBT613-midterm/blob/main/bci-midterm/notebooks/project.ipynb

This opens the notebook published in this repository. The saved Drive notebook above
contains the verified execution. To reproduce, select a CPU runtime, run all cells,
restart if installation requires it, run all again, then save the notebook with outputs.

The final run manifest must say `full_planned_cohort: true`, `n_subjects: 17`, and
`colab_verified: true` to count as Colab verification. A successful local CPU run is useful
evidence but does not establish a fresh Colab result. Do not turn that checkbox on merely
because local runtime was under 30 minutes. If server downloads exceed the limit, record
that honestly and distinguish download time from analysis time.
