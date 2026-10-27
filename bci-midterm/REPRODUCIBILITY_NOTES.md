# Reproducibility notes

## Local full-cohort execution

Python 3.12.14 on a local CPU, 17 subjects and 34 runs. The notebook completed all
32 code cells, saved 17 plot outputs, and passed its data, QC and split assertions.
The recorded analysis time with cached data was 1.1544 minutes. These are local results,
not an uncached Colab runtime claim.

## Colab CPU verification completed

Saved notebook: https://colab.research.google.com/drive/19qYoD5urhaAcdMhhokzYAZPZuN2ra2Rh

The CPU hardware option was verified in Colab's runtime dialog. The upload was approved
by the user and completed using the native macOS file picker.

The initial install on Python 3.13 required a runtime restart. The notebook stopped when
SciPy imported a new NumPy module against a previously loaded NumPy binary
(`cannot import name '_slice' from 'numpy._core.umath'`). Restarting the session and
running all cells resolved the import failure without changing the scientific pipeline.

The next run reached data preparation, downloaded subjects 8-11, then received HTTP 429
from Zenodo. The downloader now paces requests across threads, uses larger range blocks,
honors numeric or HTTP-date Retry-After instructions, and retries temporary server errors.
The regression checks verify a 429 delay followed by success, and no retry for permanent 404.

The final restarted CPU session used a new, empty dataset directory. It completed all
32 code cells over 17 subjects/34 runs, with no errors and 17 saved plots. Setup plus
analysis took 7.4094 minutes; analysis including data preparation took 7.2690 minutes.
Data preparation took 179.31 seconds. Package installation had already finished before
the restart; the timed successful run includes package checks, downloads and analysis.
All recorded native BLAS/OpenMP pools used one thread. The numerical scientific results
match the local run within floating-point tolerance; loading times differ as expected.

The downloaded cloud notebook matches the script source in every cell. Cloud CSVs and
run_manifest.json supply the report and slide numbers. results/colab_verification.json
records the notebook SHA-256 and execution checks. results/input_manifest.json records
the local retrieval’s file hashes; it is not a separate cloud input-hash audit.

The cloud browser may fail to render output frames because of its cookie settings;
notebook downloads retain the actual kernel outputs. No browser privacy setting was changed.
