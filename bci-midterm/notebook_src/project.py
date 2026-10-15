# %% [markdown]
# # Calibration-free P300: does a decoder trained on other people work on a new user?
#
# **ROBT613 Midterm Project · Team _N_** — _member 1, member 2, member 3, member 4_
#
# **Dataset:** BrainInvaders2013a (Vaineau et al., 2019), loaded with MOABB → MNE.
# **Idea:** cross-subject generalization (Challenging).
#
# **Hypothesis.** A shrinkage-LDA P300 detector trained only on *other* subjects
# (leave-one-subject-out, LOSO) reaches above-chance ROC AUC on an unseen Brain Invaders
# user, but scores lower than a subject-specific detector evaluated on the same held-out
# run — and the gap remains when both detectors get the same number of training epochs.
#
# **Why it matters.** Every P300 session normally starts with a calibration phase. If a
# generic decoder works out of the box, a user can start immediately; the Brain Invaders
# adaptive mode was built on exactly this idea (Barachant & Congedo, 2014). The question
# here is how much a *linear, in-scope* decoder loses when it skips calibration, and for whom.
#
# | Stage | Notebook source | Status |
# |---|---|---|
# | 1 · Data audit | 00, 00a, 01 | **done** (smoke-tested on subject 8) |
# | 2 · Signal inspection | 03 | **done** (smoke-tested on subject 8) |
# | 3 · Artifact handling | 04, 05 | **done** (smoke-tested on subject 8) |
# | 4 · Neurophysiological check | 01 | **done** (smoke-tested on subject 8) |
# | 5 · Features and classifier | 02 | **done** (smoke-tested on subject 8) |
# | 6 · Hypothesis test | 02, 06 | **code done** (smoke-tested on subjects 8–10) |
# | 7 · Reproducibility | 00a | partial |
#
# **Saved outputs** come from a local smoke test on subjects 8–10 (`BCI_N_SUBJECTS=3`). The
# submitted outputs must come from the full Colab run on all 17 subjects.
#
# Methods and code are adapted from the ROBT613 BCI Workshop notebooks 00–06
# (https://github.com/berdakh/ROBT613-BCI-Workshop); each reused block cites its notebook.

# %% [markdown]
# ## 0 · Setup
# Versions are pinned to the workshop's `requirements.txt` (mne 1.13.2, moabb 1.7.2).
# On Colab the first cell installs them; restart the runtime if pip asks for it.

# %%
# Setup pattern adapted from workshop notebook 01 (setup cell).
import sys, subprocess, os, time
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'mne==1.13.2', 'moabb==1.7.2'])
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

SEED = 613
T_START = time.perf_counter()          # total runtime, printed in stage 7
rng = np.random.default_rng(SEED)
DATA_ROOT = Path(os.environ.get('BCI_DATA', './mne_data')).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MNE_DATA'] = str(DATA_ROOT)
# MOABB 1.7.2 prefers the NEMAR mirror, which holds only one file per session for
# BI2013a (3.mat = Non-Adaptive Training). With the mirror, the Adaptive Training run
# would be missing without any error, so we download the full Zenodo archives instead.
os.environ['MOABB_DOWNLOAD_PROVIDER'] = 'upstream'
mne.set_log_level('WARNING')
plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
import importlib.metadata as metadata
print({p: metadata.version(p) for p in ['mne', 'moabb', 'numpy', 'scipy', 'scikit-learn']})
print('Dataset cache:', DATA_ROOT)

# %% [markdown]
# ### Study configuration
# **Subject subset.** BrainInvaders2013a has 24 subjects. Subjects 1–7 recorded eight
# sessions each, and the MOABB loader downloads all eight (≈1 GB per subject), which does
# not fit the 30-minute Colab budget. Subjects 8–24 recorded exactly one session (≈130 MB
# each, ≈2.2 GB in total), so every included subject contributes the same protocol and the
# same amount of data. We therefore use **subjects 8–24 (n = 17)**.
#
# **Runs.** Each session has a Non-Adaptive and an Adaptive run, each with a Training phase
# (fixed 80 target / 400 non-target flashes) followed by an Online phase. We load only the
# two **Training** phases: their flash counts are fixed, so every subject has the same class
# balance, and the two runs give the within-subject baseline a leave-one-run-out split.
# The run order was randomised per session (Vaineau et al., 2019), so MOABB's run name
# (file number − 1) does not tell which condition a run is; we read it from each
# subject's `meta.yml`.

# %%
SUBJECTS = list(range(8, 25))          # documented subset, see the text above
# Quick smoke test: BCI_N_SUBJECTS=1 audits only the first subject. Leave unset for the report.
AUDIT_SUBJECTS = SUBJECTS[:int(os.environ.get('BCI_N_SUBJECTS', len(SUBJECTS)))]
EVENT_ID = {'Target': 33285, 'NonTarget': 33286}
STIM_CHANNEL = 'STI 014'
# Group-level tests (Wilcoxon, Spearman) need enough subjects; below this they are skipped.
# BCI_MIN_GROUP_N=2 forces every group branch to run in a small smoke test (p-values meaningless).
MIN_GROUP_N = int(os.environ.get('BCI_MIN_GROUP_N', 6))

# %% [markdown]
# ### Shared helper functions
# All later stages call these functions instead of copying cells.

# %%
import yaml
from moabb.datasets import BI2013a


def make_dataset():
    """BrainInvaders2013a, Training phases of both runs (see configuration)."""
    return BI2013a(non_adaptive=True, adaptive=True, training=True, online=False)


def run_conditions(subject, dataset):
    """Map MOABB run name -> 'adaptive_training' etc., read from the subject's meta.yml."""
    first_file = Path(dataset.data_path(subject)[0])
    meta = yaml.safe_load((first_file.parents[1] / 'meta.yml').read_text())
    return {str(int(Path(r['filename']).stem) - 1): f"{r['experimental_condition']}_{r['type']}"
            for r in meta['runs']}


def load_subject_raws(subject, dataset=None):
    """Return a list of (session, run, condition, Raw) for one subject."""
    dataset = dataset or make_dataset()
    data = dataset.get_data(subjects=[subject])[subject]
    cond = run_conditions(subject, dataset)
    return [(sess, run, cond[run], raw) for sess, runs in data.items()
            for run, raw in runs.items()]


def count_events(raw):
    """Count Target / NonTarget flashes from the stimulus channel."""
    events = mne.find_events(raw, stim_channel=STIM_CHANNEL, shortest_event=1, verbose=False)
    codes, counts = np.unique(events[:, 2], return_counts=True)
    found = dict(zip(codes.tolist(), counts.tolist()))
    return {name: found.get(code, 0) for name, code in EVENT_ID.items()}, found, events


def band_sd(raw, l_freq=1.0, h_freq=20.0, edge_s=2.0):
    """Per-channel SD (V) in the P300 band, on a filtered copy; edges dropped.

    The amplifier is DC-coupled, so the raw SD is dominated by offsets and drift and
    says nothing about scaling. This audit-only filter does not touch `raw`;
    the real filter is chosen and justified in stage 2.
    """
    filt = raw.copy().filter(l_freq, h_freq, picks='eeg')
    edge = int(edge_s * raw.info['sfreq'])
    return filt.get_data(picks='eeg')[:, edge:-edge].std(axis=1)


def line_noise_db(raw, line=50.0):
    """Per-channel power at the mains frequency relative to 40–60 Hz neighbours (dB)."""
    psd = raw.compute_psd(picks='eeg', fmin=35, fmax=65, n_fft=1024)
    f, p = psd.freqs, psd.get_data()
    at_line = p[:, np.abs(f - line) <= 1].mean(axis=1)
    near = p[:, (np.abs(f - line) > 2) & (np.abs(f - line) < 10)].mean(axis=1)
    return 10 * np.log10(at_line / near)


def audit_raw(subject, session, run, condition, raw):
    """One row of the data-audit table for a single run."""
    eeg = raw.get_data(picks='eeg')
    sd = band_sd(raw)
    line_db = line_noise_db(raw)
    n_evt, all_codes, events = count_events(raw)
    soa_ms = np.median(np.diff(events[:, 0])) / raw.info['sfreq'] * 1e3
    return {
        'subject': subject, 'session': session, 'run': run, 'condition': condition,
        'n_eeg': len(mne.pick_types(raw.info, eeg=True)),
        'sfreq_Hz': raw.info['sfreq'],
        'duration_s': round(raw.times[-1], 1),
        'n_target': n_evt['Target'], 'n_nontarget': n_evt['NonTarget'],
        'other_codes': {c: n for c, n in all_codes.items() if c not in EVENT_ID.values()},
        'median_soa_ms': round(soa_ms, 1),
        'max_abs_dc_mV': round(float(np.abs(eeg.mean(axis=1)).max()) * 1e3, 2),
        'median_sd_1_20Hz_uV': round(float(np.median(sd)) * 1e6, 1),
        'max_sd_1_20Hz_uV': round(float(sd.max()) * 1e6, 1),
        'max_sd_channel': raw.ch_names[int(np.argmax(sd))],
        'flat_channels': [raw.ch_names[i] for i in np.where(sd < 0.5e-6)[0]],
        'max_line_50Hz_dB': round(float(line_db.max()), 1),
        'max_line_channel': raw.ch_names[int(np.argmax(line_db))],
    }

# %% [markdown]
# ## 1 · Data audit
# Provenance from the acquisition paper (Vaineau et al., 2019), checked against the loaded
# files below:
#
# | Property | Documented value | Checked in this notebook |
# |---|---|---|
# | Amplifier | g.USBamp (g.tec), OpenViBE, no digital filter | — (paper only) |
# | Sampling rate | 512 Hz | `sfreq_Hz` |
# | EEG channels | 16 wet Ag/AgCl: Fp1 Fp2 F5 AFz F6 T7 Cz T8 P7 P3 Pz P4 P8 O1 Oz O2 | `n_eeg`, `ch_names` |
# | Reference / ground | left earlobe / Fz | — (paper only) |
# | Event codes | Target = 33285, NonTarget = 33286 | `n_target`, `n_nontarget`, `other_codes` |
# | Training phase | 80 target / 400 non-target flashes per run | counts below |
# | Paradigm | 36 symbols in 12 groups; 12 flashes per repetition, 2 contain the target (1 : 5) | ratio below |
#
# **Source discrepancy.** The MOABB 1.7.2 `BI2013a` docstring names a TMSi Nexus amplifier.
# The paper and MOABB's own `METADATA` name a g.USBamp with ground at Fz. We follow the paper.
#
# **Units.** MNE stores EEG in volts, and the BI2013a loader applies no µV→V factor. The
# amplifier is DC-coupled, so the raw signal carries offsets of several mV (`max_abs_dc_mV`)
# and the raw SD says nothing about scaling. We therefore check the SD in the 1–20 Hz band
# on a filtered copy. Values of roughly 5–20 µV on posterior channels, and larger values on
# Fp1/Fp2 (blinks), mean the data are in volts. A value near 10⁶ µV would mean µV values
# stored as volts.

# %%
dataset = make_dataset()
rows, raws_by_subject, load_seconds = [], {}, {}
for subject in AUDIT_SUBJECTS:
    t0 = time.perf_counter()
    runs = load_subject_raws(subject, dataset)
    load_seconds[subject] = time.perf_counter() - t0
    raws_by_subject[subject] = runs
    rows += [audit_raw(subject, *run) for run in runs]
audit = pd.DataFrame(rows)
audit

# %%
# Totals per subject and the invariants that later stages rely on.
per_subject = audit.groupby('subject').agg(
    runs=('run', 'count'), target=('n_target', 'sum'), nontarget=('n_nontarget', 'sum'),
    median_sd_uV=('median_sd_1_20Hz_uV', 'median'))
per_subject['target_ratio'] = (per_subject['target'] / per_subject['nontarget']).round(3)
per_subject['load_s'] = pd.Series(load_seconds).round(1)
print(per_subject.to_string())
print('\nTotal epochs:', int(per_subject[['target', 'nontarget']].sum().sum()),
      '| subjects:', len(per_subject),
      '| download + load time:', round(sum(load_seconds.values()), 1), 's')

assert (audit['sfreq_Hz'] == 512).all(), 'unexpected sampling rate'
assert (audit['n_eeg'] == 16).all(), 'unexpected channel count'
# The within-subject baseline needs exactly one Adaptive and one Non-Adaptive training run.
conds = audit.groupby('subject')['condition'].apply(sorted)
assert all(c == ['adaptive_training', 'nonadaptive_training'] for c in conds), conds
assert ((audit['n_target'] == 80) & (audit['n_nontarget'] == 400)).all(), 'flash counts'
assert (audit['median_sd_1_20Hz_uV'].between(1, 100)).all(), 'amplitude scale is off'

# %% [markdown]
# ### Raw signal and sensor layout
# One subject is shown as an example. The montage comes from MOABB's standard 10-20
# positions; the plotted layout is checked against the documented channel list.

# %%
example_subject = AUDIT_SUBJECTS[0]
_, example_run, example_cond, example_raw = raws_by_subject[example_subject][0]
print(f'Subject {example_subject}, run {example_run} ({example_cond})')
print(example_raw)
print('Channels:', example_raw.ch_names)
print('Montage set:', example_raw.get_montage() is not None)

# %%
if example_raw.get_montage() is None:
    example_raw.set_montage('standard_1020')
fig = example_raw.plot_sensors(show_names=True, title=f'Sensor layout · subject {example_subject}')

# %%
_, _, example_events = count_events(example_raw)
# Unfiltered data; the viewer only removes each channel's DC offset (remove_dc=True).
# MOABB also stores every flash as an annotation; we hide them and draw the events instead.
print('Annotations:', len(example_raw.annotations), 'flashes')
fig = (example_raw.copy().set_annotations(None)
       .plot(events=example_events, event_id=EVENT_ID, start=30, duration=10, n_channels=16,
             scalings=dict(eeg=100e-6), show_scrollbars=False, remove_dc=True,
             title=f'Raw EEG · subject {example_subject}, {example_cond}'))

# %% [markdown]
# **Audit notes** _(fill in after the full run):_ subjects and runs included; any run whose
# counts differ from 80 / 400; extra event codes and what they mean; channels that look flat
# or noisy (`max_sd_channel`); total download and load time on Colab.
#
# _Smoke test on subject 8 (local run, 2026-10-15):_ two training runs, run `0` =
# adaptive and run `2` = non-adaptive, each with 80 / 400 flashes; 512 Hz; 16 EEG channels.
# DC offsets reach about 20 mV. The 1–20 Hz SD is about 9–11 µV on posterior channels and
# about 35–50 µV on Fp1/Fp2. This confirms that the data are in volts. The raw trace shows
# two blinks on Fp1/Fp2/F5/AFz/F6, which supports using Fp1/Fp2 as ocular proxies in stage 3.
# It also shows strong 50 Hz mains noise, worst on P7 at about +30 dB over the neighbouring
# frequencies, which stage 2 must handle.

# %% [markdown]
# ## 2 · Signal inspection and filter choice (notebook 03)
# The starting point is the course's P300 contract from notebook 01,
# `P300(fmin=0.5, fmax=20, resample=128)`. Each part of it has to earn its place on *this*
# recording. We check four questions on the measured spectra:
#
# 1. Is a high-pass needed, and why 0.5 Hz rather than 0.1 or 1 Hz?
# 2. Is 20 Hz the right low-pass?
# 3. Does the 50 Hz mains noise found in the audit need a notch?
# 4. Is 128 Hz safe from aliasing and lossless for the P300 band?
#
# **Rules that keep this stage leakage-free.**
# * Every run is filtered on its own, never concatenated. The Within condition tests on a
#   held-out run, so no filter support crosses from one run into another (notebook 03,
#   "A reproducible filter report").
# * The filter has no data-fitted parameters. The cut-offs were fixed before decoding and
#   are never chosen by decoding score.

# %%
# Stage-2 configuration and helpers. Filter reporting follows notebook 03.
from scipy import signal

FILTER = dict(l_freq=0.5, h_freq=20.0, method='fir', phase='zero',
              fir_window='hamming', fir_design='firwin')
SFREQ_OUT = 128.0              # new Nyquist 64 Hz, far above the 25 Hz stop-band edge
EPOCH_WINDOW = (-0.1, 0.8)     # used from stage 4 on; needed here for the edge guard


def design_filter(sfreq, l_freq=FILTER['l_freq'], h_freq=FILTER['h_freq']):
    """FIR coefficients exactly as Raw.filter builds them (same window and design)."""
    return mne.filter.create_filter(None, sfreq, l_freq, h_freq, method='fir', phase='zero',
                                    fir_window=FILTER['fir_window'],
                                    fir_design=FILTER['fir_design'], verbose=False)


def half_length_s(h, sfreq):
    """Zero-phase FIR support on each side of a sample, in seconds."""
    return (len(h) - 1) / 2 / sfreq


def filter_resample(raw, l_freq=FILTER['l_freq'], h_freq=FILTER['h_freq'], sfreq_out=SFREQ_OUT):
    """Stage-2 preprocessing of ONE run: band-pass, then resample with the events.

    Returns (raw_out, events_out). The input `raw` is not modified.
    """
    _, _, events = count_events(raw)
    filt = raw.copy().filter(l_freq, h_freq, picks='eeg', method='fir', phase='zero',
                             fir_window=FILTER['fir_window'], fir_design=FILTER['fir_design'])
    return filt.resample(sfreq_out, events=events)


def edge_guard(events, sfreq, n_times, guard_s, window=EPOCH_WINDOW):
    """True for flashes whose epoch window stays guard_s away from both recording edges."""
    t = events[:, 0] / sfreq
    return (t + window[0] >= guard_s) & (t + window[1] <= n_times / sfreq - guard_s)


def welch(raw, n_fft):
    """Welch PSD of the EEG channels in V²/Hz (MNE removes each segment's DC)."""
    spec = raw.compute_psd(method='welch', picks='eeg', fmin=0, fmax=raw.info['sfreq'] / 2,
                           n_fft=n_fft)
    return spec.freqs, spec.get_data()


def band_rms_uV(freqs, psd, lo, hi):
    """RMS amplitude (µV) per channel in [lo, hi] Hz, by integrating the PSD."""
    sel = (freqs >= lo) & (freqs <= hi)
    return np.sqrt(psd[:, sel].sum(axis=1) * (freqs[1] - freqs[0])) * 1e6


h_main = design_filter(512.0)
GUARD_S = half_length_s(h_main, 512.0)
print(f'FIR: {len(h_main)} taps = {len(h_main) / 512:.2f} s, '
      f'support ±{GUARD_S:.2f} s around each sample')

# %% [markdown]
# ### 2.1 · Raw spectrum and the filter's response
# The figure shows Welch PSDs before and after preprocessing, for the median channel and for
# P7, the noisiest channel in the audit. The right panel shows the filter's own gain. Notebook
# 03 stresses that a data spectrum is not the filter response: it is the input spectrum
# shaped by that response.

# %%
f_raw, p_raw = welch(example_raw, n_fft=4096)                 # 0.125 Hz resolution
example_out, example_events_out = filter_resample(example_raw)
f_out, p_out = welch(example_out, n_fft=1024)                 # same resolution at 128 Hz
i_p7 = example_raw.ch_names.index('P7')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
for f, p, style, name in [(f_raw, p_raw, '-', 'raw 512 Hz'),
                          (f_out, p_out, '--', 'band-passed, 128 Hz')]:
    ax.plot(f, 10 * np.log10(np.median(p, axis=0)), style, color='C0', label=f'median ch · {name}')
    ax.plot(f, 10 * np.log10(p[i_p7]), style, color='C3', label=f'P7 · {name}', alpha=0.8)
ax.axvspan(FILTER['l_freq'], FILTER['h_freq'], color='0.9', zorder=0, label='passband')
ax.set(xscale='log', xlim=(0.1, 256), xlabel='Frequency (Hz)', ylabel='PSD (dB re 1 V²/Hz)',
       title=f'Subject {example_subject}, {example_cond}')
ax.legend(fontsize=8, loc='lower left')

w, H = signal.freqz(h_main, worN=2 ** 16, fs=512.0)
gain_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))
ax = axes[1]
ax.plot(w, gain_db)
for x, lab in [(FILTER['l_freq'], '0.5'), (FILTER['h_freq'], '20'), (50, '50 Hz mains')]:
    ax.axvline(x, color='0.6', ls=':')
    ax.text(x, -115, lab, rotation=90, va='bottom', ha='right', fontsize=8)
ax.set(xscale='log', xlim=(0.05, 256), ylim=(-120, 5), xlabel='Frequency (Hz)',
       ylabel='Gain (dB)', title=f'FIR 0.5–20 Hz, {len(h_main)} taps, zero-phase')
plt.tight_layout()
plt.show()


def gain_at(freq):
    return gain_db[np.argmin(np.abs(w - freq))]


print('Filter report (notebook 03 checklist)')
filter_report = {
    'family': 'FIR, windowed design (firwin), Hamming window',
    'passband': f"{FILTER['l_freq']}–{FILTER['h_freq']} Hz",
    '-6 dB points': '0.25 Hz and 22.5 Hz (MNE auto transition bands 0.5 Hz / 5 Hz)',
    'length': f'{len(h_main)} taps = {len(h_main) / 512:.2f} s at 512 Hz',
    'phase': 'zero-phase (non-causal, offline only)',
    'gain 0.1 / 1 / 10 / 25 / 50 Hz': ' / '.join(f'{gain_at(x):.1f}' for x in [0.1, 1, 10, 25, 50]) + ' dB',
    'applied': 'continuous data, each run separately, before epoching',
    'boundaries': f"MNE 'reflect_limited' padding + edge guard of {GUARD_S:.2f} s (2.3)",
    'resampling': f'{SFREQ_OUT:.0f} Hz after the low-pass; events resampled with the data',
}
for key, value in filter_report.items():
    print(f'  {key:<32} {value}')

# %% [markdown]
# ### 2.2 · Per-run measurements behind each decision
# These are measured on every audited run, so the full Colab run gives the numbers for all
# 17 subjects:
#
# * **`drift_frac`** is the share of 0–20 Hz power that lies below 0.5 Hz. A large value is
#   why we need a high-pass.
# * **`mains_before_uV` and `mains_after_uV`** are the RMS amplitude in 49.5–50.5 Hz on the
#   worst channel, before and after preprocessing.
# * **`mains_resid_pct`** is the mains residue as a percentage of the in-band (1–20 Hz) RMS.
#   If it is far below 1%, a notch would change nothing.
# * **`resample_max_err_uV`** is the largest difference between the 128 Hz signal and every
#   4th sample of the 512 Hz filtered signal. The 512 → 128 Hz step is lossless only if this
#   is tiny next to `inband_rms_uV`.
# * **`events_kept`** checks that resampling the events preserved the 80/400 counts.
# * **`edge_dropped`** is the number of flashes whose epoch falls within the filter support
#   of a recording edge (see 2.3).

# %%
def stage2_row(subject, run, condition, raw):
    f0, p0 = welch(raw, n_fft=4096)
    out, ev_out = filter_resample(raw)
    f1, p1 = welch(out, n_fft=1024)
    inband = band_rms_uV(f1, p1, 1, 20)
    mains0, mains1 = band_rms_uV(f0, p0, 49.5, 50.5), band_rms_uV(f1, p1, 49.5, 50.5)
    # Lossless resampling check: compare with the 512 Hz filtered data at the same instants.
    ref = raw.copy().filter(picks='eeg', **FILTER).get_data(picks='eeg')[:, ::4]
    new = out.get_data(picks='eeg')
    n = min(ref.shape[1], new.shape[1]) - 64                  # skip resampling edge samples
    err = np.abs(new[:, 64:n] - ref[:, 64:n]).max() * 1e6
    keep = edge_guard(ev_out, out.info['sfreq'], out.n_times, GUARD_S)
    counts_out = {name: int((ev_out[:, 2] == code).sum()) for name, code in EVENT_ID.items()}
    return {
        'subject': subject, 'run': run, 'condition': condition,
        'drift_frac': round(float(np.median(p0[:, f0 < 0.5].sum(1) / p0[:, f0 <= 20].sum(1))), 2),
        'mains_ch': raw.ch_names[int(np.argmax(mains0))],
        'mains_before_uV': round(float(mains0.max()), 2),
        'mains_after_uV': round(float(mains1.max()), 4),
        'mains_resid_pct': round(float(np.max(mains1 / inband)) * 100, 3),
        'inband_rms_uV': round(float(np.median(inband)), 2),
        'resample_max_err_uV': round(float(err), 3),
        'events_kept': counts_out == {'Target': 80, 'NonTarget': 400},
        'edge_dropped_T': int((~keep & (ev_out[:, 2] == EVENT_ID['Target'])).sum()),
        'edge_dropped_NT': int((~keep & (ev_out[:, 2] == EVENT_ID['NonTarget'])).sum()),
    }


stage2 = pd.DataFrame([stage2_row(s, run, cond, raw)
                       for s, runs in raws_by_subject.items() for _, run, cond, raw in runs])
print(stage2.to_string(index=False))
assert stage2['events_kept'].all(), 'resampling changed the event counts'

# %% [markdown]
# ### 2.3 · Recording edges: how close to an edge is still safe?
# A zero-phase FIR needs samples on both sides. Near the start or end of a recording, MNE pads
# the missing side by reflection, and the output there is distorted. Some runs end only about
# 2 s after the last flash, so we measure this instead of assuming it.
#
# The test: we cut the example run at 200 s, filter both the cut and the full version, and
# compare them before the cut. The full version is the reference, because there it has real
# data on both sides.

# %%
CUT_S = 200.0
full = example_raw.copy().filter(picks='eeg', **FILTER).get_data(picks='eeg')
cut = example_raw.copy().crop(0, CUT_S).filter(picks='eeg', **FILTER).get_data(picks='eeg')
n_cut = cut.shape[1]
edge_err = np.abs(cut - full[:, :n_cut]).max(axis=0) * 1e6       # worst channel, µV
dist_s = (n_cut - 1 - np.arange(n_cut)) / 512.0
show = dist_s <= 6

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.semilogy(dist_s[show], np.maximum(edge_err[show], 1e-3))
ax.axvline(GUARD_S, color='C3', ls='--', label=f'filter support {GUARD_S:.2f} s = guard')
ax.axhline(1.0, color='0.6', ls=':', label='1 µV')
ax.invert_xaxis()
ax.set(xlabel='Distance before the recording edge (s)', ylabel='Worst-channel error (µV)',
       title='Edge distortion of the 0.5–20 Hz FIR (truncation test)')
ax.legend()
plt.show()

# How many flashes would each high-pass cut-off lose to its own edge guard?
hp_rows = []
for l_freq in [0.1, 0.5, 1.0]:
    g = half_length_s(design_filter(512.0, l_freq=l_freq), 512.0)
    lost = sum(int((~edge_guard(count_events(raw)[2], 512.0, raw.n_times, g)).sum())
               for runs in raws_by_subject.values() for *_, raw in runs)
    hp_rows.append({'high-pass (Hz)': l_freq, 'guard (s)': round(g, 2),
                    'flashes lost to edges (all audited runs)': lost})
print(pd.DataFrame(hp_rows).to_string(index=False))

# %% [markdown]
# ### 2.4 · Is the passband right for the P300?
# We test each edge of the band on the example subject, using the target − non-target
# difference at Pz:
#
# * **High-pass.** A high-pass can shrink a slow, positive component such as the P300. We
#   compare 0.1, 0.5, 1 and 2 Hz with the low-pass fixed at 20 Hz.
# * **Low-pass.** We compare 20 and 30 Hz with the high-pass fixed at 0.5 Hz. If the
#   20–30 Hz band carried ERP information, the two waves would differ in a systematic way.
#
# **One factor at a time.** Every curve uses the *same* flashes: one common edge guard taken
# from the longest filter (0.1 Hz, ±16.5 s). As in the pipeline, the curves carry no
# baseline correction (see 3.3).
#
# **What the zero level is matters.** We report each peak two ways:
# * measured from 0 µV;
# * measured from the −100–0 ms mean.
#
# Neither reference is clean. With no baseline, slow offsets left by gentle high-passes stay
# in the wave. The pre-flash window is not neutral either: 3.3 shows that it holds a target's
# response before about 20% of non-targets but before almost no targets. The truth for the
# P300 amplitude lies somewhere between the two columns.
#
# The band **0.5–20 Hz was fixed beforehand** from the course contract. These checks only
# confirm that it does not distort the waveform. We do **not** pick the band that maximises
# the class difference, because that would be tuning on labels.

# %%
HP_CANDIDATES = [0.1, 0.5, 1.0, 2.0]
GUARD_COMMON = max(half_length_s(design_filter(512.0, l_freq=l), 512.0) for l in HP_CANDIDATES)


def pz_difference(runs, l_freq, h_freq=FILTER['h_freq'], guard_s=GUARD_COMMON):
    """Target − non-target average at Pz (µV), both runs pooled, same flashes for every band."""
    diffs, n_kept = [], 0
    for *_, raw in runs:
        out, ev = filter_resample(raw, l_freq=l_freq, h_freq=h_freq)
        ev = ev[edge_guard(ev, out.info['sfreq'], out.n_times, guard_s)]
        n_kept += len(ev)
        ep = mne.Epochs(out, ev, EVENT_ID, *EPOCH_WINDOW, baseline=None, picks=['Pz'],
                        preload=True, verbose=False)
        diffs.append(ep['Target'].average().data[0] - ep['NonTarget'].average().data[0])
    return ep.times, np.mean(diffs, axis=0) * 1e6, n_kept


def peak_in_window(t, wave, lo=0.2, hi=0.6):
    """Peak in [lo, hi] s: (from 0 µV, from the −100–0 ms mean, latency ms)."""
    win = (t >= lo) & (t <= hi)
    i = np.argmax(wave[win])
    return wave[win][i], wave[win][i] - wave[t < 0].mean(), int(round(t[win][i] * 1e3))


runs_example = raws_by_subject[example_subject]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
peak_rows = []
for l_freq in HP_CANDIDATES:
    t, dw, n_kept = pz_difference(runs_example, l_freq)
    from_zero, from_pre, lat = peak_in_window(t, dw)
    chosen = l_freq == FILTER['l_freq']
    axes[0].plot(t * 1e3, dw, lw=2.5 if chosen else 1.2,
                 label=f'{l_freq} Hz' + (' (chosen)' if chosen else ''))
    peak_rows.append({'high-pass (Hz)': l_freq, 'flashes': n_kept,
                      'peak from 0 (µV)': round(from_zero, 2),
                      'peak from pre-flash (µV)': round(from_pre, 2), 'latency (ms)': lat})
axes[0].set(title=f'High-pass (low-pass 20 Hz) · Pz, subject {example_subject}',
            xlabel='Time after flash (ms)', ylabel='Target − non-target (µV)')
axes[0].legend(title='high-pass', fontsize=8)

lp_waves = {}
for h_freq in [20.0, 30.0]:
    t, dw, n_kept = pz_difference(runs_example, FILTER['l_freq'], h_freq=h_freq)
    lp_waves[h_freq] = dw
    axes[1].plot(t * 1e3, lp_waves[h_freq], lw=2.5 if h_freq == FILTER['h_freq'] else 1.2,
                 label=f'{h_freq:.0f} Hz' + (' (chosen)' if h_freq == FILTER['h_freq'] else ''))
axes[1].set(title='Low-pass (high-pass 0.5 Hz)', xlabel='Time after flash (ms)')
axes[1].legend(title='low-pass', fontsize=8)
for ax in axes:
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
plt.tight_layout()
plt.show()

peaks = pd.DataFrame(peak_rows)
for col in ['peak from 0 (µV)', 'peak from pre-flash (µV)']:
    peaks[col.split(' (')[0] + ' % of 0.1 Hz'] = (100 * peaks[col] / peaks[col].iloc[0]).round(0)
print(peaks.to_string(index=False))
extra = lp_waves[30.0] - lp_waves[20.0]
print(f'\n20 vs 30 Hz low-pass: RMS of the difference between the two waves = '
      f'{np.sqrt(np.mean(extra ** 2)):.2f} µV, against an RMS of {np.sqrt(np.mean(lp_waves[20.0] ** 2)):.2f} µV '
      f'for the 20 Hz wave itself.')

# %% [markdown]
# ### Stage 2 decisions
# The numbers quoted below come from the local smoke test on subject 8. The table in 2.2
# gives them for every subject in the full run.
#
# | Decision | Evidence |
# |---|---|
# | **High-pass 0.5 Hz** (−6 dB at 0.25 Hz) | 73–80% of the 0–20 Hz power lies below 0.5 Hz (`drift_frac`). Without a high-pass, drift dominates every epoch. The 0.1 Hz alternative needs a 33 s filter, and its ±16.5 s edge guard costs 34 flashes on subject 8, against 3 at 0.5 Hz (2.3). In 2.4 every cut-off uses the same 926 flashes. Relative to 0.1 Hz, the 0.5 Hz peak keeps 90% measured from 0 µV and 102% measured from the pre-flash mean, at the same latency (391 ms). So the loss is between 0 and about 10%, depending on the reference. At 1 Hz it keeps 76–94%. At 2 Hz it keeps 61–73%, the largest peak moves to the earlier component (242 ms), and undershoots appear. |
# | **Low-pass 20 Hz** (−6 dB at 22.5 Hz, −52 dB at 25 Hz) | The course P300 contract (notebooks 01–02) uses 20 Hz. In 2.4, moving the low-pass to 30 Hz changes the difference wave by 0.21 µV RMS, about 7% of the wave's own RMS (3.13 µV), and adds only small ripples. The 20–30 Hz band therefore carries almost no ERP information here. Cutting it also removes the out-of-band P7 noise seen in 2.1. |
# | **FIR rather than IIR** | Notebook 03: a symmetric FIR has linear phase, and applied zero-phase it leaves ERP latencies unchanged. A windowed FIR design is also always stable. An IIR reaches a sharp response with fewer coefficients, but its phase is non-linear unless filtered forward–backward, and it needs care to stay numerically stable. The FIR's cost is its length (6.6 s), which the edge guard handles. |
# | **No notch filter** | The 50 Hz noise on P7 drops from ≈61–83 µV RMS to ≈0.01–0.03 µV, at most 0.16% of the in-band RMS (`mains_resid_pct`). The low-pass already gives −75 dB at 50 Hz, so a notch would add nothing (notebook 03, Practice 3). |
# | **Resample to 128 Hz** | The new Nyquist is 64 Hz, far above the 25 Hz stop band, so nothing can alias. The largest difference from the 512 Hz signal is ≈0.04 µV, against an in-band RMS of ≈9 µV. The 80/400 event counts survive, and event timing moves by at most 1/256 s (3.9 ms). The 4× fewer samples make ICA in stage 3 cheaper. |
# | **Edge guard ±3.30 s** | In the truncation test the error exceeds 1 µV only within about 2.8 s of an edge. Stage 5 drops any flash whose epoch lies within the filter support of an edge, and reports the count (`edge_dropped_*`: 3 of 960 flashes on subject 8). |
#
# **P7.** Its high-frequency noise lies entirely outside the passband. Its in-band SD was
# normal in the audit, so it is **not** marked bad on spectral grounds. Stage 3 re-checks it
# after filtering.
#
# **Limitation.** The zero-phase filter needs future samples, so it suits only this offline
# study. A causal linear-phase FIR of the same length would delay the signal by 3.3 s; an
# online BCI would need a causal IIR or minimum-phase design and would see a different
# waveform.

# %% [markdown]
# ## 3 · Artifact handling (notebook 04) and quality control (notebook 05)
# The audit found blinks on the frontal channels. There are no EOG channels, so **Fp1 and
# Fp2 serve as ocular proxies**: they are the scalp sites closest to the eyes and carry the
# largest blink deflections. This is a declared approximation, not a measured EOG.
#
# **Order of operations, per run (no data crosses runs):**
#
# 1. **Flat-channel check** on the stage-2 data. A flat channel is marked bad and kept out of
#    the ICA fit.
# 2. **ICA**, configured as in notebook 04: FastICA, 15 components, `random_state=SEED`,
#    `max_iter=1000`. It is fitted on a **1–20 Hz copy at 128 Hz**, because a 1 Hz high-pass
#    stabilises ICA. The unmixing is then applied to the 0.5–20 Hz data, which has the same
#    channels and reference.
# 3. **Component exclusion needs two independent kinds of evidence** (notebook 04: never
#    remove a component by its index or its variance):
#    * its time course is an outlier in correlation with Fp1/Fp2 (MNE `find_bads_eog`,
#      z ≥ 3);
#    * its scalp map is frontal: at least 50% of the map's power sits on Fp1, Fp2 and AFz.
# 4. **Noisy-channel check** after ICA. A non-proxy channel is marked bad if its SD is more
#    than 3× or less than ⅓× the median SD of the 14 non-proxy channels, which is ≥9× off in
#    power. A bad channel is marked bad, ignored by the amplitude rejection, and **removed
#    from the features of every subject** in stage 5 (`GLOBAL_BADS`), because LOSO needs the
#    same channel set for every subject. This is channel selection, which is in scope.
#    Interpolation is mentioned in notebook 04 but never demonstrated in notebooks 00–06, so
#    we avoid it. A run with more than 2 bad channels is flagged for manual review.
#    *Why a ratio rather than a robust z-score?* On subject 8 the 14 channels are so alike
#    that the MAD is tiny. A robust z-score then flagged P7 and F6 at z ≈ 3.5, although their
#    SD was only 1.3× the median. The ratio rule only fires for real outliers.
# 5. **Epochs** from −100 to 800 ms with **no baseline correction** (see 3.3). First the
#    edge guard from stage 2 is applied, then a **peak-to-peak rejection at 100 µV** on any
#    EEG channel (see 3.4). The two kinds of drop are counted separately, by class
#    (notebook 05).
#
# **Reference.** We keep the recorded left-earlobe reference. Notebook 04 shows that a common
# average spreads any large artifact into every channel. With 16 sparse channels, two of them
# blink-dominated proxies, the average is a poor estimate of zero. The one-sided earlobe
# reference may add a small lateral asymmetry; we state this as a limitation.
#
# **Leakage.** ICA, channel checks and thresholds use no labels. Each run is cleaned only with
# its own data, so the held-out run in the Within condition never shapes the cleaning of the
# training run. All thresholds are fixed here, before any decoding. Labels appear in this
# stage only in *descriptive* checks (3.2 and 3.3), never in a choice.

# %%
ICA_PARAMS = dict(n_components=15, method='fastica', random_state=SEED, max_iter=1000)  # notebook 04
ICA_FIT_BAND = (1.0, 20.0)
EOG_PROXIES = ['Fp1', 'Fp2']
EOG_Z = 3.0
FRONTAL = ['Fp1', 'Fp2', 'AFz']
FRONTAL_MIN = 0.5
FLAT_UV = 0.5
BAD_RATIO = 3.0
MAX_BAD = 2
REJECT_P2P_UV = 100.0
BASELINE = None


def fit_ica(raw, main, bads):
    """Fit ICA on a 1–20 Hz, 128 Hz copy of one run; return (ica, evidence table, excluded)."""
    fit = raw.copy().filter(*ICA_FIT_BAND, picks='eeg', method='fir', phase='zero',
                            fir_window=FILTER['fir_window'], fir_design=FILTER['fir_design'])
    fit.resample(SFREQ_OUT)
    fit.info['bads'] = list(bads)
    ica = mne.preprocessing.ICA(**ICA_PARAMS).fit(fit, picks='eeg')
    eog_idx, scores = ica.find_bads_eog(fit, ch_name=EOG_PROXIES, threshold=EOG_Z,
                                        measure='zscore')
    maps = ica.get_components()
    names = [ch for ch in fit.copy().pick('eeg', exclude='bads').ch_names]
    rows = [r for r, ch in enumerate(names) if ch in FRONTAL]
    frontal = (maps[rows] ** 2).sum(0) / (maps ** 2).sum(0)
    evidence = pd.DataFrame({
        'component': np.arange(ica.n_components_),
        **{f'r_{ch}': np.round(s, 2) for ch, s in zip(EOG_PROXIES, scores)},
        'eog_outlier': np.isin(np.arange(ica.n_components_), eog_idx),
        'frontal_frac': np.round(frontal, 2),
    })
    evidence['excluded'] = evidence['eog_outlier'] & (evidence['frontal_frac'] >= FRONTAL_MIN)
    excluded = evidence.loc[evidence['excluded'], 'component'].tolist()
    return ica, evidence, excluded, fit


def concat_epochs(epochs_list):
    """Concatenate runs whose bad-channel lists differ: all copies get the union of bads."""
    bads = sorted({ch for ep in epochs_list for ch in ep.info['bads']})
    copies = []
    for ep in epochs_list:
        ep = ep.copy()
        ep.info['bads'] = list(bads)
        copies.append(ep)
    return mne.concatenate_epochs(copies, verbose=False)


def blink_onsets_s(raw):
    """Blink times (s) detected on the ocular proxies (MNE find_eog_events)."""
    blinks = mne.preprocessing.find_eog_events(raw, ch_name=EOG_PROXIES, verbose=False)
    return blinks[:, 0] / raw.info['sfreq']


def clean_run(subject, run, condition, raw):
    """Stage 2 + stage 3 for ONE run. Returns a dict with clean data, epochs and a QC row."""
    main, events = filter_resample(raw)
    sd = main.get_data(picks='eeg').std(axis=1) * 1e6
    eeg_names = [main.ch_names[i] for i in mne.pick_types(main.info, eeg=True)]
    flat = [ch for ch, s in zip(eeg_names, sd) if s < FLAT_UV]

    ica, evidence, excluded, fit = fit_ica(raw, main, flat)
    var_removed = (ica.get_explained_variance_ratio(fit, components=excluded, ch_type='eeg')['eeg']
                   if excluded else 0.0)
    clean = main.copy()
    clean.info['bads'] = list(flat)
    ica.apply(clean, exclude=excluded)

    good = [ch for ch in eeg_names if ch not in EOG_PROXIES and ch not in flat]
    sd_clean = clean.get_data(picks=good).std(axis=1)
    ratio = sd_clean / np.median(sd_clean)
    noisy = [ch for ch, r in zip(good, ratio) if r > BAD_RATIO or r < 1 / BAD_RATIO]
    bads = flat + noisy
    review = len(bads) > MAX_BAD
    clean.info['bads'] = bads          # kept in the data; rejection ignores bad channels

    # Descriptive: blinks inside epoch windows, by class (before cleaning).
    t_blink = blink_onsets_s(main)
    t_flash = events[:, 0] / SFREQ_OUT
    has_blink = np.array([np.any((t_blink >= t + EPOCH_WINDOW[0]) & (t_blink <= t + EPOCH_WINDOW[1]))
                          for t in t_flash])
    is_target = events[:, 2] == EVENT_ID['Target']

    metadata = pd.DataFrame({'subject': subject, 'run': run, 'condition': condition,
                             'flash': np.arange(len(events)),
                             'label': np.where(is_target, 'Target', 'NonTarget'),
                             'blink': has_blink})
    keep = edge_guard(events, SFREQ_OUT, clean.n_times, GUARD_S)
    epochs = mne.Epochs(clean, events[keep], EVENT_ID, *EPOCH_WINDOW, baseline=BASELINE,
                        picks=eeg_names, metadata=metadata[keep].reset_index(drop=True),
                        preload=True, verbose=False)   # all 16 channels in every run
    n_before = {lab: int((epochs.metadata['label'] == lab).sum()) for lab in EVENT_ID}
    epochs.drop_bad(reject={'eeg': REJECT_P2P_UV * 1e-6}, verbose=False)
    n_after = {lab: int((epochs.metadata['label'] == lab).sum()) for lab in EVENT_ID}

    qc = {
        'subject': subject, 'run': run, 'condition': condition,
        'bad_channels': ','.join(bads) or '-', 'review': review,
        'max_sd_ratio': f'{ratio.max():.2f} ({good[int(np.argmax(ratio))]})',
        'ica_excluded': ','.join(map(str, excluded)) or '-',
        'ica_var_removed': round(float(var_removed), 2),
        'blinks_per_min': round(len(t_blink) / (clean.times[-1] / 60), 1),
        'blink_frac_T': round(float(has_blink[is_target].mean()), 3),
        'blink_frac_NT': round(float(has_blink[~is_target].mean()), 3),
        'edge_drop_T': int((~keep & is_target).sum()), 'edge_drop_NT': int((~keep & ~is_target).sum()),
        'amp_drop_T': n_before['Target'] - n_after['Target'],
        'amp_drop_NT': n_before['NonTarget'] - n_after['NonTarget'],
        'kept_T': n_after['Target'], 'kept_NT': n_after['NonTarget'],
    }
    return dict(main=main, clean=clean, events=events, ica=ica, evidence=evidence,
                excluded=excluded, epochs=epochs, qc=qc)


t0 = time.perf_counter()
cleaned = {(s, run): clean_run(s, run, cond, raw)
           for s, runs in raws_by_subject.items() for _, run, cond, raw in runs}
qc = pd.DataFrame([c['qc'] for c in cleaned.values()])
print(f'Stage 3 for {len(cleaned)} runs: {time.perf_counter() - t0:.1f} s\n')
print(qc.to_string(index=False))
assert not qc['review'].any(), 'a run needs manual channel review (see bad_channels)'
GLOBAL_BADS = sorted({ch for c in cleaned.values() for ch in c['clean'].info['bads']})
print('\nChannels to drop from the features of every subject (stage 5):', GLOBAL_BADS or 'none')
CORE_P300 = ['Cz', 'Pz', 'P3', 'P4', 'Oz']
# Dropping a core P300 site for all 17 subjects because of one run is a decision for a person,
# not for this rule. If either assert fires, keep it and send the message back for review.
assert not set(GLOBAL_BADS) & set(CORE_P300), f'core P300 channel flagged bad: {GLOBAL_BADS}'
assert len(GLOBAL_BADS) <= 2, f'too many channels would be dropped for everyone: {GLOBAL_BADS}'

# %% [markdown]
# ### 3.1 · Evidence for the excluded components (example run)
# The table lists each component's correlation with the proxies, whether MNE flags it as an
# outlier, and the frontal share of its map. Next come the scalp maps (excluded components are
# named in the printout). The last plot shows the average blink before and after removal, which
# is "what ICA removed" (notebook 04).

# %%
ex = cleaned[(example_subject, example_run)]
print(ex['evidence'].to_string(index=False))
print('Excluded:', ex['excluded'])
fig = ex['ica'].plot_components(title=f'ICA maps · subject {example_subject}, {example_cond}',
                                show=False)
plt.show()

blink_evoked = mne.preprocessing.create_eog_epochs(ex['main'], ch_name=EOG_PROXIES,
                                                   verbose=False).average()
fig = ex['ica'].plot_overlay(blink_evoked, exclude=ex['excluded'], picks='eeg',
                             title='Average blink: before (red) vs after (black) ICA', show=False)
plt.show()

# %% [markdown]
# ### 3.2 · Eye activity is a confound here, and ICA removes it without removing the P300
# If blinks are time-locked to targets, an uncleaned decoder could detect targets from the
# eyes rather than from the brain. The QC table above already counts this (`blink_frac_T`
# versus `blink_frac_NT`). Below, the target − non-target difference at Fp1 (ocular) and Pz
# (P300) is shown before and after ICA. This is a descriptive check of the cleaning; it chooses
# nothing.

# %%
def difference_wave(raw, events, ch):
    keep = edge_guard(events, SFREQ_OUT, raw.n_times, GUARD_S)
    ep = mne.Epochs(raw, events[keep], EVENT_ID, *EPOCH_WINDOW, baseline=None, picks=[ch],
                    preload=True, verbose=False)
    return ep.times, (ep['Target'].average().data[0] - ep['NonTarget'].average().data[0]) * 1e6


runs_ex = [cleaned[k] for k in cleaned if k[0] == example_subject]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
conf_rows = []
for ax, ch in zip(axes, ['Fp1', 'Pz']):
    for key, style in [('main', '--'), ('clean', '-')]:
        waves = [difference_wave(c[key], c['events'], ch) for c in runs_ex]
        t, dw = waves[0][0], np.mean([w[1] for w in waves], axis=0)
        ax.plot(t * 1e3, dw, style, label='before ICA' if key == 'main' else 'after ICA')
        win = (t >= 0.2) & (t <= 0.6)
        conf_rows.append({'channel': ch, 'data': 'before ICA' if key == 'main' else 'after ICA',
                          'T−NT max 200–600 ms (µV)': round(dw[win].max(), 2)})
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set(title=f'{ch} · subject {example_subject}', xlabel='Time after flash (ms)')
axes[0].set_ylabel('Target − non-target (µV)')
axes[0].legend()
plt.tight_layout()
plt.show()
print(pd.DataFrame(conf_rows).to_string(index=False))

# Group-level test of the blink confound (needs several subjects; run on Colab).
blink_subj = qc.groupby('subject')[['blink_frac_T', 'blink_frac_NT']].mean()
if len(blink_subj) >= MIN_GROUP_N:
    from scipy.stats import wilcoxon
    stat = wilcoxon(blink_subj['blink_frac_T'], blink_subj['blink_frac_NT'])
    print(f'\nBlink fraction T vs NT across {len(blink_subj)} subjects: '
          f'median T {blink_subj.blink_frac_T.median():.3f}, NT {blink_subj.blink_frac_NT.median():.3f}, '
          f'Wilcoxon p = {stat.pvalue:.4f}')
else:
    print(f'\nOnly {len(blink_subj)} subject(s) audited: the group blink test runs on the full set.')

# %% [markdown]
# ### 3.3 · Why there is no baseline correction
# Notebook 05 warns that a baseline is only fair if the pre-event interval is neutral and equal
# across conditions. In Brain Invaders the flashes are dense: the median interval is about
# 350–400 ms, shorter than one epoch. The −100–0 ms window of a flash therefore falls 250–400 ms
# after the previous flash, inside that flash's P300 window. The table shows how often the
# previous flash was a target, for each class.

# %%
prev_rows = []
for (s, run), c in cleaned.items():
    is_t = c['events'][:, 2] == EVENT_ID['Target']
    prev_t = np.r_[False, is_t[:-1]]
    soa_ms = np.diff(c['events'][:, 0]) / SFREQ_OUT * 1e3
    prev_rows.append({'subject': s, 'run': run,
                      'P(prev=Target | Target)': round(prev_t[is_t].mean(), 3),
                      'P(prev=Target | NonTarget)': round(prev_t[~is_t].mean(), 3),
                      'interval p5/p50/p95 (ms)': '/'.join(f'{v:.0f}' for v in np.percentile(soa_ms, [5, 50, 95]))})
print(pd.DataFrame(prev_rows).to_string(index=False))

# %% [markdown]
# The previous flash is a target before roughly one non-target in five but almost never before
# a target, so the two classes would get **systematically different baselines**. Subtracting
# them would add a class-dependent offset that is not part of the response to the current
# flash. We therefore use no baseline. Stage 2's 0.5 Hz high-pass already removes the slow
# offsets a baseline would otherwise handle.

# %% [markdown]
# ### 3.4 · Rejection threshold: sensitivity by class
# After ICA, peak-to-peak amplitudes are far smaller. The threshold of **100 µV** is meant to
# catch only gross residual artifacts. The curve shows the retained fraction for each class
# over a range of thresholds (notebook 05, Practice 4) on the same candidate epochs. Unequal
# curves would signal a selection effect.

# %%
def candidate_epochs(key):
    """Edge-guarded, unrejected epochs of every run on all 16 channels (bads kept, flagged)."""
    return concat_epochs([mne.Epochs(c[key], c['events'][edge_guard(c['events'], SFREQ_OUT, c[key].n_times, GUARD_S)],
                                     EVENT_ID, *EPOCH_WINDOW, baseline=None,
                                     picks=mne.pick_types(c[key].info, eeg=True, exclude=[]),
                                     preload=True, verbose=False)
                          for c in cleaned.values()])


cand = candidate_epochs('clean')
p2p_uV = np.ptp(cand.get_data(picks='eeg'), axis=2).max(axis=1) * 1e6     # good channels only
labels = cand.events[:, 2]
thresholds = np.arange(40, 301, 10)
fig, ax = plt.subplots(figsize=(7, 3.5))
for name, code in EVENT_ID.items():
    ax.plot(thresholds, [(p2p_uV[labels == code] <= th).mean() for th in thresholds], 'o-',
            ms=3, label=name)
ax.axvline(REJECT_P2P_UV, color='C3', ls='--', label=f'{REJECT_P2P_UV:.0f} µV (chosen)')
ax.set(ylim=(0, 1.02), xlabel='Peak-to-peak threshold (µV)', ylabel='Retained fraction',
       title='After ICA · same candidate epochs, all audited runs')
ax.legend()
plt.show()
print('Epoch max peak-to-peak after ICA (µV): p50 / p90 / p99 / max =',
      ' / '.join(f'{v:.0f}' for v in np.percentile(p2p_uV, [50, 90, 99, 100])))

# The same threshold on the same flashes *before* ICA, for comparison.
cand_pre = candidate_epochs('main')
p2p_pre = np.ptp(cand_pre.get_data(picks='eeg'), axis=2).max(axis=1) * 1e6
for tag, amp in [('before ICA', p2p_pre), ('after ICA', p2p_uV)]:
    over = amp > REJECT_P2P_UV
    print(f'{tag:>10}: {over.sum()} of {len(amp)} epochs exceed {REJECT_P2P_UV:.0f} µV '
          f'(Target {over[labels == EVENT_ID["Target"]].mean():.1%}, '
          f'NonTarget {over[labels == EVENT_ID["NonTarget"]].mean():.1%})')

# %%
# Final epoch collection for stages 4–6: one Epochs object per run, metadata keeps the groups.
epochs_by_run = {key: c['epochs'] for key, c in cleaned.items()}
totals = qc[['edge_drop_T', 'edge_drop_NT', 'amp_drop_T', 'amp_drop_NT', 'kept_T', 'kept_NT']].sum()
print(totals.to_string())
print(f"Retained: {100 * (totals.kept_T + totals.kept_NT) / (80 + 400) / len(qc):.1f}% of flashes")

# %% [markdown]
# ### Stage 3 decisions
# The numbers quoted below come from the local smoke test on subject 8. The QC table gives
# them for every run in the full run.
#
# | Decision | Evidence |
# |---|---|
# | **ICA, FastICA with 15 components, fitted per run on 1–20 Hz** | Notebook 04 configuration. A fit takes under 1 s per run. |
# | **Exclusion rule: Fp1/Fp2 correlation outlier AND frontal map** | Run 0 of subject 8 excludes two components. Component 0 has r = 0.94 / 0.81 with Fp1/Fp2 and puts 83% of its map power on Fp1/Fp2/AFz. Component 1 has r = −0.29 / −0.55, is a z-outlier, and is 95% frontal. Run 2 excludes one blink component, 85% frontal. Component 6 in run 0 is 89% frontal but not correlated with the proxies, so the AND rule keeps it. The average blink disappears (3.1). The excluded components carry ≈77% of the variance of the ICA training copy, which is dominated by blinks. |
# | **ICA removes the eye confound** | Subject 8 blinks in 18–33% of target epochs against 12–19% of non-target epochs. Before ICA, the target − non-target difference at Fp1 peaks at 22.8 µV around 580 ms, after the P300, which is the timing of a post-target blink. After ICA it is 4.6 µV, while the Pz difference is unchanged (8.16 → 8.16 µV, both runs pooled). |
# | **No bad channels for subject 8** | No channel is flat. After ICA the largest SD ratio is 1.30 (P7), far below 3. A first robust z-score rule flagged P7 at z ≈ 3.5 for a 1.3× SD, which is why we switched to the ratio rule. |
# | **No baseline** | P(previous flash = target) is 1–4% before targets and ≈20% before non-targets (3.3), so a baseline would differ by class. |
# | **Peak-to-peak 100 µV** | After ICA, p99 = 79 µV. 100 µV removes 4 of 957 epochs (Target 0.6%, NonTarget 0.4%), and the class curves overlap (3.4). Before ICA the same threshold would remove 227 of 957 epochs: 35.8% of targets against 21.3% of non-targets. That would be a class-biased selection, which is another reason to clean before rejecting. |
#
# **Subjects 9–10 (smoke test).**
# * Subject 9's run 0 has F6 at 3.53× the median SD, so F6 is marked bad and joins
#   `GLOBAL_BADS`. Stage 5 then drops it for every subject, which is the policy working as
#   designed.
# * Subject 9 blinks rarely (4–9 per minute) and not around targets (1–3% of epochs in
#   either class).
# * Subject 10 shows the same target-locked blinking as subject 8 (30–41% of target epochs
#   against 13–19% of non-target epochs).
# * Pooled over the 3 subjects, 100 µV removes 7 of 2874 epochs after ICA. Before ICA it
#   would remove 462, with 25.3% of targets against 14.2% of non-targets.
#
# **Limitation.** ICA uses the whole run, including samples after each flash, so it is an
# offline method. Proxy-based detection can miss eye movements that are weak on Fp1/Fp2 (for
# example horizontal saccades), and residual ocular activity remains on Fp1/Fp2 (≈5 µV in the
# difference wave). Stage 5 should decide, before decoding, whether Fp1/Fp2 enter the
# features.

# %% [markdown]
# ## 4 · Neurophysiological check (notebook 01)
# Before any decoding, we show that the brain response we intend to decode is present: a
# positive **target − non-target** difference over parietal sites. The check runs per subject
# and for the group.
#
# * **Input.** `epochs_by_run` from stage 3: ICA-cleaned, no baseline, edge-guarded, 100 µV
#   rejection. The two training runs of each subject are pooled.
# * **Predeclared window: 250–600 ms at Pz.** This is the window notebook 01 marks for the
#   P300, and it matches the 240–600 ms range given in the dataset paper (Vaineau et al.,
#   2019). It was fixed before looking at these data, not picked where the difference looks
#   largest.
# * **Per-subject test.** The window-mean difference at Pz is compared against a
#   label-permutation null (2000 shuffles, one-sided: P300 > 0). Flashes overlap in time
#   (interval < epoch length), so epochs are not independent. We read this p-value as a
#   descriptive check, not an exact one.
# * **Group test.** A one-sided Wilcoxon signed-rank test of the per-subject window means
#   against 0. It needs ≥ 6 subjects, so it runs in the full Colab run.
# * **Inputs for the secondary question.** For each subject we record peak amplitude, peak
#   latency, and the correlation of the subject's difference wave with the grand average of
#   the *other* subjects. Later these will be compared with how well the LOSO decoder does
#   for that subject. This is descriptive and never feeds a decoder.

# %%
ERP_CHANNELS = ['Cz', 'Pz', 'P3', 'P4', 'Oz']
P300_WINDOW = (0.25, 0.60)        # notebook 01 predeclared window; Vaineau et al. 2019: 240–600 ms
N_PERM = 2000


def subject_epochs(subject):
    """Both training runs of one subject, concatenated (same 16 channels)."""
    return concat_epochs([ep for (s, _), ep in epochs_by_run.items() if s == subject])


def window_mean(ep, ch, window=P300_WINDOW):
    """Mean amplitude (V) of one channel in the window, one value per epoch."""
    sel = (ep.times >= window[0]) & (ep.times < window[1])
    return ep.get_data(picks=[ch])[:, 0, sel].mean(axis=1)


def permutation_p(values, is_target, n_perm=N_PERM, seed=SEED):
    """One-sided label-permutation p-value for mean(target) − mean(non-target) > 0."""
    rng_local = np.random.default_rng(seed)
    observed = values[is_target].mean() - values[~is_target].mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng_local.permutation(is_target)
        null[i] = values[perm].mean() - values[~perm].mean()
    return observed, (np.sum(null >= observed) + 1) / (n_perm + 1)


erp = {}
for subject in AUDIT_SUBJECTS:
    ep = subject_epochs(subject)
    ev_t, ev_nt = ep['Target'].average(), ep['NonTarget'].average()
    erp[subject] = dict(epochs=ep, target=ev_t, nontarget=ev_nt,
                        diff=mne.combine_evoked([ev_t, ev_nt], weights=[1, -1]))

# No interpolation (stage 3 policy): a channel bad in any subject drops out of the group average.
grand_diff = mne.grand_average([e['diff'] for e in erp.values()], interpolate_bads=False, drop_bads=True)
times = grand_diff.times
win = (times >= P300_WINDOW[0]) & (times < P300_WINDOW[1])

rows = []
for subject, e in erp.items():
    pz = e['diff'].get_data(picks=['Pz'])[0] * 1e6
    i_peak = np.argmax(pz[win])
    values = window_mean(e['epochs'], 'Pz') * 1e6
    is_t = (e['epochs'].metadata['label'] == 'Target').to_numpy()
    observed, p = permutation_p(values, is_t)
    others = [o['diff'] for s, o in erp.items() if s != subject]
    if others:
        grand_others = mne.grand_average(others, interpolate_bads=False, drop_bads=True)
        chans = [ch for ch in ERP_CHANNELS if ch in grand_others.ch_names]
        template = grand_others.get_data(picks=chans)[:, win].ravel()
        mine = e['diff'].get_data(picks=chans)[:, win].ravel()
        similarity = float(np.corrcoef(mine, template)[0, 1])
    else:
        similarity = np.nan
    rows.append({'subject': subject, 'n_T': int(is_t.sum()), 'n_NT': int((~is_t).sum()),
                 'Pz_window_mean_uV': round(observed, 2), 'perm_p': round(p, 4),
                 'Pz_peak_uV': round(float(pz[win][i_peak]), 2),
                 'Pz_peak_ms': int(round(times[win][i_peak] * 1e3)),
                 'similarity_to_others': round(similarity, 2)})
erp_table = pd.DataFrame(rows)
print(erp_table.to_string(index=False))

if len(erp_table) >= MIN_GROUP_N:
    from scipy.stats import wilcoxon
    res = wilcoxon(erp_table['Pz_window_mean_uV'], alternative='greater')
    print(f'\nGroup: {np.sum(erp_table.perm_p < 0.05)}/{len(erp_table)} subjects with perm p < .05; '
          f'Wilcoxon (window mean > 0) p = {res.pvalue:.2g}')
else:
    print(f'\nOnly {len(erp_table)} subject(s) audited: the group test runs on the full set.')

# %% [markdown]
# ### 4.1 · Class averages and their difference (example subject)
# The figure shows the target and non-target averages, and their difference, at the five
# planned sites. The grey band is the predeclared window.

# %%
e = erp[example_subject]
fig, axes = plt.subplots(1, len(ERP_CHANNELS), figsize=(16, 3.4), sharey=True)
for ax, ch in zip(axes, ERP_CHANNELS):
    for ev, name, style in [(e['target'], 'Target', '-'), (e['nontarget'], 'NonTarget', '-'),
                            (e['diff'], 'T − NT', '--')]:
        ax.plot(ev.times * 1e3, ev.get_data(picks=[ch])[0] * 1e6, style, label=name)
    ax.axvspan(*(np.array(P300_WINDOW) * 1e3), color='0.9', zorder=0)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set(title=ch, xlabel='Time (ms)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend(fontsize=8)
fig.suptitle(f'Subject {example_subject} · n = {len(e["epochs"]["Target"])} targets, '
             f'{len(e["epochs"]["NonTarget"])} non-targets', y=1.03)
plt.show()

# %% [markdown]
# ### 4.2 · Single-trial view at Pz (targets only)
# An average can hide whether the response is consistent or driven by a few trials
# (notebook 01, Practice 3). The ERP image stacks every target epoch at Pz; a vertical band
# in the window means the P300 appears trial after trial.

# %%
fig = e['epochs']['Target'].plot_image(picks=['Pz'], sigma=1.0, show=False,
                                       title=f'Pz, target epochs · subject {example_subject}')
plt.show()

# %% [markdown]
# ### 4.3 · Group difference wave and its topography
# Each subject's Pz difference wave appears as a thin line, with the grand average in bold.
# The topographies show the grand-average difference at fixed latencies. Fp1/Fp2 are
# included so that any residual frontal (ocular) difference after ICA stays visible.

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
for subject, e_s in erp.items():
    ax.plot(times * 1e3, e_s['diff'].get_data(picks=['Pz'])[0] * 1e6, color='0.6', lw=0.8)
ax.plot(times * 1e3, grand_diff.get_data(picks=['Pz'])[0] * 1e6, color='C3', lw=2.5,
        label=f'grand average (n = {len(erp)})')
ax.axvspan(*(np.array(P300_WINDOW) * 1e3), color='0.9', zorder=0)
ax.axhline(0, color='0.5', lw=0.8)
ax.set(xlabel='Time after flash (ms)', ylabel='Target − non-target at Pz (µV)',
       title='Per-subject (grey) and grand-average difference waves')
ax.legend()
plt.show()

fig = grand_diff.plot_topomap(times=[0.15, 0.25, 0.35, 0.45, 0.55], ch_type='eeg',
                              time_unit='ms', show=False)
fig.suptitle('Grand-average target − non-target', y=1.05)
plt.show()

# %% [markdown]
# ### Stage 4 findings
# These come from the local smoke test on subject 8. The table and the group test above give
# them for all subjects in the full run.
#
# * **The P300 is present.** Subject 8 keeps 158 target and 795 non-target epochs. The
#   window-mean difference at Pz (250–600 ms) is **+3.14 µV**, with permutation p = 0.0005,
#   the smallest value 2000 shuffles can give.
# * **Shape.** There are two positive peaks, at ≈240 ms and ≈390 ms (the largest: 8.15 µV at
#   Pz, 391 ms), over Cz, Pz, P3 and P4. An earlier negativity near 170 ms appears at Oz and
#   P4. The topography at 250–350 ms is centro-parietal, the classic P300 distribution.
# * **Consistency.** The ERP image at Pz shows positive bands near 250 ms and 400 ms running
#   through many single target epochs, so the average is not carried by a few trials.
# * **The first peak sits at the window's edge.** The ≈240 ms peak lies partly before the
#   predeclared 250 ms start. We keep the window as declared, because moving it after
#   seeing the data would be tuning. The decoder in stage 5 uses the whole 0–800 ms
#   waveform, so this has no effect on decoding.
# * **Subjects 9 and 10 (smoke test).** Both show the P300 as well, with window means of
#   4.95 µV and 2.94 µV, both at p = 0.0005. Their peaks are 9.12 µV at 406 ms and 7.84 µV
#   at 336 ms. Each subject's difference wave correlates 0.65–0.69 with the average of the
#   other two.
# * **What this does not show.** A clear average difference does not guarantee single-trial
#   decodability, and it says nothing yet about cross-subject transfer. Those are the
#   questions for stages 5–6.

# %% [markdown]
# ## 5 · Features and classifier (notebook 02)
# Every choice below is fixed **before** the first decoding result is seen, and none is
# tuned afterwards.
#
# | Choice | Value | Reason |
# |---|---|---|
# | Features | Mean amplitude in **16 consecutive 50 ms windows, 0–800 ms**, per channel | The window-mean method of notebooks 01–02 with finer windows. This is the "decimated waveform at ~20 Hz" declared in the README before any data were seen. Stage 4 shows two peaks (≈240 and ≈390 ms), which the four wide windows of notebook 02 would blur together. |
# | Channels | **14 minus `GLOBAL_BADS`**: all EEG except Fp1/Fp2 and any channel flagged in stage 3 (13 in the 3-subject smoke test, where F6 is dropped) | Fp1/Fp2 served as ocular proxies for ICA. They keep ≈5 µV of residual target-related difference, and blinks are more frequent after targets (stage 3). Leaving them out stops the decoder from reading the eyes. The cost: frontal brain activity at those two sites is lost. `GLOBAL_BADS` is pooled over all subjects, the test subject included. This is a label-free channel-quality flag that keeps one channel set for LOSO, and we state it as the one piece of test-subject information (no labels, no epochs) that shapes training. |
# | Pipeline | `FunctionTransformer(window means)` → `StandardScaler` → `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` | One scikit-learn `Pipeline` (notebook 02), so the scaling and the covariance shrinkage are learned on training folds only. |
# | Hyper-parameters | **none tuned** | The shrinkage is set analytically (Ledoit–Wolf, `shrinkage='auto'`) inside each training fold. Windows and channels are fixed above. So no nested CV is needed, and no test data can leak into a choice. |
# | Within-subject CV | `LeaveOneGroupOut` with group = run (2 folds) | One run trains, the other tests. Epochs from one run never land on both sides, and filtering and ICA were already done per run. A random epoch split would leak, because overlapping flashes share samples. |
# | Metrics | **ROC AUC** (primary), balanced accuracy | Classes are 1 : 5. AUC ranks the continuous LDA scores. Balanced accuracy uses the default decision rule, with LDA priors from training prevalence (notebook 02), and no threshold tuning. |
# | Chance level | `permutation_test_score`, 200 label shuffles **within each run** | This is the empirical null for AUC under the same CV; the smallest possible p is 1/201 ≈ 0.005. |
#
# Stage 6 reuses this exact pipeline for the LOSO conditions. This stage gives the
# subject-specific (**Within**) baseline.

# %%
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, permutation_test_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay

FEATURE_WINDOWS = [(round(a, 2), round(a + 0.05, 2)) for a in np.arange(0.0, 0.8, 0.05)]
EPOCH_CHANNELS = next(iter(epochs_by_run.values())).ch_names          # the 16 EEG channels
FEATURE_CHANNELS = [ch for ch in EPOCH_CHANNELS if ch not in EOG_PROXIES + GLOBAL_BADS]
EPOCH_TIMES = next(iter(epochs_by_run.values())).times
assert all(((EPOCH_TIMES >= a) & (EPOCH_TIMES < b)).sum() >= 5 for a, b in FEATURE_WINDOWS)
N_PERM_DECODE = 200


def erp_window_means(X, times=EPOCH_TIMES, windows=FEATURE_WINDOWS):
    """(n_epochs, n_channels, n_times) → (n_epochs, n_channels × n_windows); stateless."""
    feats = [X[:, :, (times >= a) & (times < b)].mean(axis=-1) for a, b in windows]
    return np.concatenate(feats, axis=1)


PIPE = Pipeline([
    ('erp_windows', FunctionTransformer(erp_window_means)),
    ('scale', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])


def to_xy(epochs_list):
    """Stack epochs → X (n, 14, t), y (1 = Target), metadata with subject / run groups."""
    X = np.concatenate([ep.get_data(picks=FEATURE_CHANNELS) for ep in epochs_list])
    meta = pd.concat([ep.metadata for ep in epochs_list], ignore_index=True)
    y = (meta['label'] == 'Target').to_numpy().astype(int)
    assert len(X) == len(y) == len(meta)
    return X, y, meta


print(f'{len(FEATURE_CHANNELS)} channels: {FEATURE_CHANNELS}')
print(f'{len(FEATURE_WINDOWS)} windows of 50 ms → {len(FEATURE_CHANNELS) * len(FEATURE_WINDOWS)} features')
X_demo, y_demo, _ = to_xy([epochs_by_run[(example_subject, example_run)]])
print('Feature matrix for one run:', PIPE[:1].fit_transform(X_demo).shape)

# %% [markdown]
# ### 5.1 · What the features carry (example subject, descriptive)
# The heat map shows the target − non-target difference of each feature (channel × window).
# It shows where the information sits. It is not used to select features.

# %%
X_s, y_s, meta_s = to_xy([ep for (s, _), ep in epochs_by_run.items() if s == example_subject])
# Features are ordered window-major: [w0: ch0..ch13, w1: ch0..ch13, ...] → (n, window, channel).
F = erp_window_means(X_s).reshape(len(X_s), len(FEATURE_WINDOWS), len(FEATURE_CHANNELS)).transpose(0, 2, 1)
# Cross-check against a direct window mean at Pz in 300–350 ms.
i_pz, i_w = FEATURE_CHANNELS.index('Pz'), FEATURE_WINDOWS.index((0.3, 0.35))
direct = X_s[:, FEATURE_CHANNELS.index('Pz'), (EPOCH_TIMES >= 0.3) & (EPOCH_TIMES < 0.35)].mean(-1)
assert np.allclose(F[:, i_pz, i_w], direct)
diff_map = (F[y_s == 1].mean(0) - F[y_s == 0].mean(0)) * 1e6
fig, ax = plt.subplots(figsize=(10, 4))
lim = np.abs(diff_map).max()
im = ax.imshow(diff_map, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim)
ax.set(yticks=range(len(FEATURE_CHANNELS)), yticklabels=FEATURE_CHANNELS,
       xticks=range(len(FEATURE_WINDOWS)), xticklabels=[f'{int(a * 1e3)}' for a, _ in FEATURE_WINDOWS],
       xlabel='Window start (ms)', title=f'Target − non-target per feature (µV) · subject {example_subject}')
plt.colorbar(im, ax=ax, label='µV')
plt.show()

# %% [markdown]
# ### 5.2 · Within-subject decoding (leave-one-run-out) with a permutation chance level

# %%
def within_subject(subject, n_perm=N_PERM_DECODE):
    """LORO decoding for one subject: per-test-run scores + permutation p for AUC."""
    X, y, meta = to_xy([ep for (s, _), ep in epochs_by_run.items() if s == subject])
    groups = meta['run'].to_numpy()
    logo = LeaveOneGroupOut()
    cv = cross_validate(clone(PIPE), X, y, groups=groups, cv=logo,
                        scoring={'auc': 'roc_auc', 'bacc': 'balanced_accuracy'})
    test_runs = [np.unique(groups[test])[0] for _, test in logo.split(X, y, groups)]
    auc_obs, null, p = permutation_test_score(clone(PIPE), X, y, groups=groups, cv=logo,
                                              scoring='roc_auc', n_permutations=n_perm,
                                              random_state=SEED)
    folds = pd.DataFrame({'subject': subject, 'test_run': test_runs,
                          'auc': cv['test_auc'], 'bacc': cv['test_bacc']})
    summary = {'subject': subject, 'auc_mean': cv['test_auc'].mean(), 'auc_sd': cv['test_auc'].std(ddof=1),
               'bacc_mean': cv['test_bacc'].mean(), 'perm_p': p,
               'null_95': np.percentile(null, 95), 'null_mean': null.mean()}
    return folds, summary, null


t0 = time.perf_counter()
within_folds, within_rows, nulls = [], [], {}
for subject in AUDIT_SUBJECTS:
    folds, summary, null = within_subject(subject)
    within_folds.append(folds)
    within_rows.append(summary)
    nulls[subject] = null
within_folds = pd.concat(within_folds, ignore_index=True)
within = pd.DataFrame(within_rows).round(3)
print(f'Within-subject decoding, {len(within)} subject(s): {time.perf_counter() - t0:.1f} s\n')
print(within_folds.round(3).to_string(index=False))
print()
print(within.to_string(index=False))
print(f"\nAcross subjects: AUC {within.auc_mean.mean():.3f} ± "
      f"{within.auc_mean.std(ddof=1) if len(within) > 1 else float('nan'):.3f} (mean ± SD), "
      f"{(within.perm_p < 0.05).sum()}/{len(within)} subjects above chance (perm p < .05)")

# %%
# Permutation null vs observed AUC, and the confusion matrix at the default threshold.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
row = within.set_index('subject').loc[example_subject]
axes[0].hist(nulls[example_subject], bins=25, color='0.7', label='shuffled labels (within run)')
axes[0].axvline(row.auc_mean, color='C3', lw=2, label=f'observed {row.auc_mean:.3f}')
axes[0].axvline(row.null_95, color='0.3', ls='--', label=f'95th pct of null {row.null_95:.3f}')
axes[0].set(xlabel='ROC AUC (leave-one-run-out)', ylabel='count',
            title=f'Subject {example_subject}: chance level by permutation')
axes[0].legend(fontsize=8)

X_s, y_s, meta_s = to_xy([ep for (s, _), ep in epochs_by_run.items() if s == example_subject])
pred = np.empty_like(y_s)
for train, test in LeaveOneGroupOut().split(X_s, y_s, meta_s['run']):
    pred[test] = clone(PIPE).fit(X_s[train], y_s[train]).predict(X_s[test])
ConfusionMatrixDisplay.from_predictions(y_s, pred, display_labels=['NonTarget', 'Target'],
                                        normalize='true', ax=axes[1], colorbar=False)
axes[1].set_title('Default threshold, row-normalised')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Stage 5 findings
# These come from the local smoke test on subjects 8–10. The tables above give all 17
# subjects in the full run.
#
# * **Features.** 13 channels × 16 windows = 208 features, because subject 9's F6 is in
#   `GLOBAL_BADS`. A training run has ≈478 epochs.
# * **Within-subject decoding works for all three.** Leave-one-run-out AUC is 0.930 ± 0.001,
#   0.927 ± 0.018 and 0.944 ± 0.014, or **0.934 ± 0.009** across subjects. Balanced accuracy
#   is 0.80–0.84.
# * **It is far above chance.** The within-run shuffles give null means of 0.498–0.502 and
#   95th percentiles of 0.538–0.560. Every subject beats every shuffle (p = 0.005, the
#   smallest possible).
# * **At the default threshold** subject 8's non-targets are recognised 94% of the time and
#   targets 71%. The training priors (1 : 5) shift the rule toward non-target (notebook 02).
#   No threshold was tuned.
# * **What carries the information (5.1, subject 8).** It sits in the 200–450 ms windows over
#   Cz, P3, Pz and P4, with a weaker frontal contribution and an early occipital negativity
#   near 150 ms. Windows before 150 ms are near zero, which is a useful leakage check.
# * **The channel set depends on who is audited.** In a one-subject run (no F6 dropped)
#   subject 8 scored 0.925 with 14 channels; here it scores 0.930 with 13. This is the
#   label-free cross-subject dependency of `GLOBAL_BADS` stated in the table above.
# * **Bug caught.** An earlier heat map read the window-major feature order as channel-major.
#   The classifier was unaffected; the plot now asserts that it matches a direct window mean.
# * **Limitations.** With two runs, each subject's SD comes from only 2 folds. Neighbouring
#   epochs overlap, so they are not independent.

# %% [markdown]
# ## 6 · Hypothesis test: calibration-free (LOSO) vs subject-specific (Within)
# For every subject *s* and each of its two training runs *r* as the **test run**:
#
# | Condition | Trained on | Tested on |
# |---|---|---|
# | **Within** (subject-specific) | the other run of *s* | run *r* of *s* |
# | **LOSO** (calibration-free) | all runs of all other subjects | run *r* of *s* |
# | **LOSO-matched** (control) | a random subset of the other subjects' epochs with the *same* number of targets and non-targets as Within; mean of 10 draws | run *r* of *s* |
#
# Only the training data change between conditions. The test epochs, features, pipeline (stage
# 5) and metric are identical. LOSO-matched separates **whose** data train the model from
# **how much** data it gets.
#
# **Pre-declared hypotheses** (from the topic registration):
# * **H1.** LOSO is above chance on unseen subjects. Per subject and run, we shuffle the
#   *test* labels against the fixed LOSO scores 1000 times. This needs no refit, because the
#   LOSO model never saw the test subject. At group level, a one-sided Wilcoxon test checks
#   that LOSO AUC exceeds 0.5.
# * **H2.** Within > LOSO, by a one-sided paired Wilcoxon test across subjects.
# * **H3.** Within > LOSO-matched, by a one-sided paired Wilcoxon test. If the gap survives
#   equal training sizes, it comes from the *source* of the data, not its amount.
# * **Descriptive.** LOSO vs LOSO-matched, two-sided: how much more data from other people
#   helps.
#
# **Unit of analysis.** One value per subject: the mean over its two test runs. The two runs
# of a subject are not independent observations. Effect sizes are reported as the median
# paired difference and the matched-pairs rank-biserial correlation. The rubric asks whether
# a difference exceeds fold-to-fold variability, so the paired gap is also compared with the
# within-subject spread across the two test runs.
#
# **Leakage checks in code.**
# * No epoch of the test subject enters LOSO or LOSO-matched training (assert).
# * The scaler and the shrinkage are fitted inside each training set.
# * The test subject gets no normalisation of its own: a calibration-free decoder would not
#   have that.

# %%
from scipy.stats import wilcoxon, rankdata, spearmanr

N_MATCHED_DRAWS = 10
N_PERM_TEST = 1000

# The window-mean step is stateless, so computing it once per run is equivalent to running
# it inside the pipeline; the scaler + LDA part is still fitted per training set.
FEAT = {}
for key, ep in epochs_by_run.items():
    X, y, _ = to_xy([ep])
    FEAT[key] = (erp_window_means(X), y)
CLF = Pipeline(PIPE.steps[1:])                      # StandardScaler → shrinkage LDA


def stack(keys):
    return (np.concatenate([FEAT[k][0] for k in keys]),
            np.concatenate([FEAT[k][1] for k in keys]))


def auc_and_perm_p(scores, y, n_perm=N_PERM_TEST, seed=SEED):
    """AUC from ranks and a one-sided p from shuffling test labels against fixed scores."""
    r = rankdata(scores)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    rng_p = np.random.default_rng(seed)
    pos = np.argsort(rng_p.random((n_perm, len(y))), axis=1)[:, :n1]
    null = (r[pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc, (np.sum(null >= auc) + 1) / (n_perm + 1)


t0 = time.perf_counter()
subjects_6 = sorted({s for s, _ in FEAT})
rows6 = []
for s in subjects_6:
    own = sorted(k for k in FEAT if k[0] == s)
    others = [k for k in FEAT if k[0] != s]
    assert s not in {k[0] for k in others}                     # no test-subject epochs
    X_lo, y_lo = stack(others)
    loso_model = clone(CLF).fit(X_lo, y_lo)                    # one LOSO model per subject
    pos_lo, neg_lo = np.where(y_lo == 1)[0], np.where(y_lo == 0)[0]
    for test_key in own:
        X_te, y_te = FEAT[test_key]
        X_w, y_w = stack([k for k in own if k != test_key])
        auc_w = roc_auc_score(y_te, clone(CLF).fit(X_w, y_w).decision_function(X_te))
        auc_l, p_l = auc_and_perm_p(loso_model.decision_function(X_te), y_te)
        n_t, n_nt = int(y_w.sum()), int(len(y_w) - y_w.sum())
        matched = []
        for k in range(N_MATCHED_DRAWS):
            r_k = np.random.default_rng([SEED, s, int(test_key[1]), k])
            pick = np.concatenate([r_k.choice(pos_lo, n_t, replace=False),
                                   r_k.choice(neg_lo, n_nt, replace=False)])
            m = clone(CLF).fit(X_lo[pick], y_lo[pick])
            matched.append(roc_auc_score(y_te, m.decision_function(X_te)))
        rows6.append({'subject': s, 'test_run': test_key[1],
                      'n_train_within': len(y_w), 'n_train_loso': len(y_lo),
                      'within': auc_w, 'loso_matched': np.mean(matched),
                      'loso_matched_sd': np.std(matched, ddof=1), 'loso': auc_l, 'loso_perm_p': p_l})
runs6 = pd.DataFrame(rows6)
print(f'Stage 6 for {len(subjects_6)} subjects: {time.perf_counter() - t0:.1f} s\n')
print(runs6.round(3).to_string(index=False))

# Consistency: Within here must equal the stage-5 cross_validate result on the same folds.
check = runs6.merge(within_folds, on=['subject', 'test_run'])
assert len(check) == len(runs6) and np.allclose(check['within'], check['auc']), 'Within mismatch'

# %%
per_subject6 = runs6.groupby('subject')[['within', 'loso_matched', 'loso']].mean()
per_subject6['loso_runs_p<.05'] = runs6.groupby('subject')['loso_perm_p'].apply(lambda p: int((p < 0.05).sum()))
per_subject6['within_fold_sd'] = runs6.groupby('subject')['within'].std(ddof=1)
print(per_subject6.round(3).to_string())
print('\nAcross subjects (mean ± SD of per-subject AUC):')
for col, name in [('within', 'Within'), ('loso_matched', 'LOSO-matched'), ('loso', 'LOSO')]:
    sd = per_subject6[col].std(ddof=1) if len(per_subject6) > 1 else float('nan')
    print(f'  {name:<13} {per_subject6[col].mean():.3f} ± {sd:.3f}')


def paired(a, b, alternative):
    d = np.asarray(a) - np.asarray(b)
    ranks = rankdata(np.abs(d))
    rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    p = wilcoxon(d, alternative=alternative).pvalue
    return {'n': len(d), 'median_diff': np.median(d), 'mean_diff': d.mean(),
            'rank_biserial': rb, 'p': p}


ps = per_subject6
tests = pd.DataFrame({
    'H1: LOSO > 0.5': paired(ps['loso'], np.full(len(ps), 0.5), 'greater'),
    'H2: Within > LOSO': paired(ps['within'], ps['loso'], 'greater'),
    'H3: Within > LOSO-matched': paired(ps['within'], ps['loso_matched'], 'greater'),
    'LOSO vs LOSO-matched (2-sided)': paired(ps['loso'], ps['loso_matched'], 'two-sided'),
}).T
print()
print(tests.round(4).to_string())
print(f"\nFold-to-fold spread of Within (median SD over the two test runs): "
      f"{ps['within_fold_sd'].median():.3f}; median Within − LOSO gap: "
      f"{(ps['within'] - ps['loso']).median():.3f}")
if len(ps) < MIN_GROUP_N:
    print(f'\nSMOKE TEST with n = {len(ps)} subjects: a one-sided Wilcoxon test cannot reach '
          f'p < .05 below n = 5 (minimum p = {0.5 ** len(ps):.3f} here). Interpret only the full run.')

# %% [markdown]
# ### 6.1 · Per-subject view
# Each line is one subject: subject-specific, then calibration-free with matched data, then
# calibration-free with all data. The dashed line marks chance.

# %%
cond_order = ['within', 'loso_matched', 'loso']
cond_names = ['Within', 'LOSO-matched', 'LOSO']
fig, ax = plt.subplots(figsize=(6.5, 4))
for s, row in ps.iterrows():
    ax.plot(range(3), row[cond_order], '-o', color='0.6', ms=4, lw=1)
    ax.text(2.06, row['loso'], str(s), fontsize=7, va='center', color='0.4')
ax.plot(range(3), ps[cond_order].median(), '-o', color='C3', lw=2.5, ms=7,
        label=f'median (n = {len(ps)})')
ax.axhline(0.5, color='0.3', ls='--', lw=1, label='chance')
ax.set(xticks=range(3), xticklabels=cond_names, ylabel='ROC AUC (mean of 2 test runs)',
       xlim=(-0.3, 2.4), title='Same test runs, different training data')
ax.legend(loc='lower left')
plt.show()

# %% [markdown]
# ### 6.2 · Secondary question: whom does LOSO fail?
# Per subject, the LOSO AUC is compared with three things from stage 4: the P300 peak
# amplitude, the peak latency, and how similar the subject's difference wave is to the
# grand average of the *other* subjects. Spearman correlations are printed only with
# ≥ 6 subjects. This analysis is exploratory: it explains results, it is not a test of H1–H3.

# %%
expl = ps.join(erp_table.set_index('subject')[['Pz_peak_uV', 'Pz_peak_ms', 'similarity_to_others']])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
for ax, col in zip(axes, ['similarity_to_others', 'Pz_peak_uV', 'Pz_peak_ms']):
    ax.scatter(expl[col], expl['loso'], color='C0')
    for s, row in expl.iterrows():
        ax.annotate(str(s), (row[col], row['loso']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    title = col
    if len(expl) >= MIN_GROUP_N:
        rho, p = spearmanr(expl[col], expl['loso'])
        title += f'\nSpearman ρ = {rho:.2f}, p = {p:.3f}'
    ax.set(xlabel=col, title=title)
axes[0].set_ylabel('LOSO AUC')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Stage 6 findings (smoke test, subjects 8–10 — **not** the result)
# With 3 subjects, LOSO trains on only **2** other people (≈1910 epochs), not 16. A one-sided
# Wilcoxon test cannot go below p = 0.125. These numbers check that the code works and point
# in a direction; the full Colab run decides H1–H3.
#
# | Subject | Within | LOSO-matched | LOSO | LOSO perm p (both runs) |
# |---|---|---|---|---|
# | 8 | 0.930 | 0.740 | 0.779 | 0.001, 0.001 |
# | 9 | 0.927 | 0.750 | 0.784 | 0.001, 0.001 |
# | 10 | 0.944 | 0.743 | 0.806 | 0.001, 0.001 |
#
# * **H1 direction holds.** A decoder that never saw the test subject still detects targets
#   on all 6 test runs (AUC 0.73–0.83, each above every one of 1000 label shuffles).
# * **H2 direction holds.** Calibration-free decoding is clearly worse than subject-specific
#   decoding. The median gap of 0.143 AUC is ten times the fold-to-fold spread of Within
#   (median SD 0.014).
# * **H3 direction holds, and it is the more informative result.** With the *same* number of
#   training epochs, other people's data give 0.74, against 0.93 from the subject's own data.
#   So the gap comes from whose data, not how much. Adding more people's data helps
#   somewhat (LOSO 0.78–0.81 vs LOSO-matched ≈0.74), and the full run with 16 training
#   subjects will show how far that goes.
# * **Consistency check.** Within AUCs computed here equal the stage-5 `cross_validate`
#   results exactly (assert). Precomputing the stateless features changes nothing.
# * **Secondary question (6.2).** It needs the full set; 3 points cannot support a
#   correlation.
# * **Runtime.** Stage 6 for 3 subjects takes 0.6 s. With 17 subjects the LOSO fits are larger
#   (≈15k epochs), but there are still no permutation refits, so it stays well within budget.

# %% [markdown]
# ## 7 · Reproducibility checklist
# - [x] Package versions pinned and printed (setup cell)
# - [x] Random seed fixed (`SEED = 613`)
# - [x] Subject IDs listed (`SUBJECTS`) and the subset justified
# - [x] Shared code in functions
# - [ ] Full run on a fresh Colab CPU runtime under 30 minutes (see the timer below)

# %%
elapsed_min = (time.perf_counter() - T_START) / 60
print(f'Total runtime: {elapsed_min:.1f} min for {len(AUDIT_SUBJECTS)} subject(s) '
      f'({"within" if elapsed_min < 30 else "OVER"} the 30-minute budget)')
print({p: metadata.version(p) for p in ['mne', 'moabb', 'numpy', 'scipy', 'scikit-learn']})

# %% [markdown]
# ## References
# 1. E. Vaineau, A. Barachant, A. Andreev, P. C. Rodrigues, G. Cattan, M. Congedo,
#    "Brain Invaders adaptive versus non-adaptive P300 brain-computer interface dataset,"
#    arXiv:1904.09111, 2019.
# 2. A. Barachant, M. Congedo, "A plug & play P300 BCI using information geometry,"
#    arXiv:1409.0107, 2014.
# 3. V. Jayaram, A. Barachant, "MOABB: trustworthy algorithm benchmarking for BCIs,"
#    J. Neural Eng., vol. 15, no. 6, 066011, 2018.
# 4. B. Aristimunha et al., "Mother of all BCI Benchmarks," Zenodo,
#    doi:10.5281/zenodo.10034223 (software; both entries from moabb.neurotechx.com/docs/cite.html).
#
# _TODO (problem significance, 10%): add 1–2 papers on cross-subject / zero-training P300
# decoding after reading them; do not cite anything the team has not opened._
#
# ## AI tools disclosure
# _List the AI assistants used and what for (debugging, API lookup, plotting)._
