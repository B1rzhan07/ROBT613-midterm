// Build the 12-minute presentation (output/presentation.pptx) and speaker notes
// from exported notebook results. Every number comes from results/presentation_data.json.
//
//   npm install          (once, from bci-midterm/)
//   node tools/build_presentation.mjs
//
// The PDF is exported from PowerPoint (File > Export > PDF) to output/pdf/presentation.pdf.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const require = createRequire(import.meta.url);
const pptxgen = require('pptxgenjs');
const sizeOf = require('image-size');

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const d = JSON.parse(fs.readFileSync(path.join(ROOT, 'results/presentation_data.json'), 'utf8'));
const m = d.manifest, n = m.n_subjects, qc = m.qc_totals;

// Palette: deep ink for structure, coral for the P300 / transfer accent.
const INK = '1A2238', CORAL = 'E4572E', AMBER = 'E09B2D', INDIGO = '3E5C9A';
const SLATE = '5B6B82', MIST = 'EEF1F6', LINE = 'D5DBE5', WHITE = 'FFFFFF', PALE = 'C9D1E0', DEEP = '252F4A';
const COND = {within: INDIGO, loso_matched: AMBER, loso: CORAL};
const HEAD = 'Cambria', BODY = 'Calibri';
// Chart values are AUC × 1000 shown through a literal-dot format, so labels read 0.875
// regardless of the presenter's locale decimal separator.
const AUC_FMT = '0"."000';

const f3 = x => Number(x).toFixed(3);
const sci = p => { // 2.29e-5 -> runs "2.3 × 10" + superscript "−5"
  if (p >= 0.001) return [{text: p.toFixed(4)}];
  const e = Math.floor(Math.log10(p)), mant = p / 10 ** e;
  return [{text: `${mant.toFixed(1)} × 10`}, {text: `−${-e}`, options: {superscript: true}}];
};
const fig = name => path.join(ROOT, 'results/figures', name);

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Calibration-free P300 decoding';
pres.author = 'Birzhan Zhunusbekov, Arman Abakov, Nazerke Khamzina, Bekmurat Amangeldiyev';
pres.theme = {headFontFace: HEAD, bodyFontFace: BODY};

pres.defineSlideMaster({
  title: 'CONTENT', background: {color: WHITE},
  slideNumber: {x: 12.35, y: 6.98, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 11, color: SLATE, align: 'right'},
});
pres.defineSlideMaster({
  title: 'DARK', background: {color: INK},
  slideNumber: {x: 12.35, y: 6.98, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 11, color: PALE, align: 'right'},
});

const notes = [];
const STAGES = ['Audit', 'Signal', 'Artifacts', 'P300', 'Decoder', 'Evaluation', 'Hypothesis'];

function txt(s, text, o) {
  s.addText(text, {isTextBox: true, fontFace: BODY, color: INK, margin: 0, valign: 'top', ...o});
}

function card(s, x, y, w, h, fill = MIST) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x, y, w, h, fill: {color: fill}, line: {color: fill}, rectRadius: 0.08});
}

// Pipeline tracker: the seven assignment stages, current one in coral.
function tracker(s, k) {
  const x0 = 8.55, step = 0.62, y = 0.42, dia = 0.36;
  s.addShape(pres.shapes.LINE, {x: x0 + dia / 2, y: y + dia / 2, w: step * 6, h: 0, line: {color: LINE, width: 1.5}});
  STAGES.forEach((name, i) => {
    const on = i === k;
    s.addShape(pres.shapes.OVAL, {x: x0 + i * step, y, w: dia, h: dia,
      fill: {color: on ? CORAL : i < k ? INK : MIST}, line: {color: on ? CORAL : i < k ? INK : LINE, width: 1}});
    txt(s, String(i + 1), {x: x0 + i * step, y, w: dia, h: dia, fontSize: 11, bold: true, align: 'center', valign: 'middle',
      color: on || i < k ? WHITE : SLATE});
  });
}

function slide(title, kicker, note, {stage = -1, dark = false} = {}) {
  const s = pres.addSlide({masterName: dark ? 'DARK' : 'CONTENT'});
  txt(s, kicker.toUpperCase(), {x: 0.6, y: 0.4, w: 7.6, h: 0.3, fontSize: 12, bold: true, color: CORAL, charSpacing: 2});
  txt(s, title, {x: 0.6, y: 0.74, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true, color: dark ? WHITE : INK});
  if (stage >= 0) tracker(s, stage);
  s.addNotes(note);
  notes.push({title, note});
  return s;
}

function caption(s, text, color = SLATE) {
  txt(s, text, {x: 0.6, y: 6.92, w: 11.2, h: 0.32, fontSize: 11, italic: true, color});
}

// Place an image inside a box without distortion; returns the placed rectangle.
function image(s, name, x, y, w, h, align = 'center') {
  const {width, height} = sizeOf(fig(name));
  const r = Math.min(w / width, h / height), iw = width * r, ih = height * r;
  const ix = align === 'left' ? x : x + (w - iw) / 2;
  s.addImage({path: fig(name), x: ix, y: y + (h - ih) / 2, w: iw, h: ih, altText: 'Executed notebook figure'});
  return {x: ix, y: y + (h - ih) / 2, w: iw, h: ih};
}

function stat(s, x, y, w, value, label, {color = INK, size = 36, labelColor = SLATE, h = 1.4} = {}) {
  txt(s, value, {x, y, w, h: size / 60, fontFace: HEAD, fontSize: size, bold: true, color});
  txt(s, label, {x, y: y + size / 60 + 0.05, w, h: h - size / 60 - 0.05, fontSize: 14, color: labelColor});
}

function numbered(s, x, y, w, items, {fill = CORAL, color = INK, size = 15, gap = 0.62} = {}) {
  items.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, {x, y: y + i * gap + 0.02, w: 0.32, h: 0.32, fill: {color: fill}, line: {color: fill}});
    txt(s, String(i + 1), {x, y: y + i * gap + 0.02, w: 0.32, h: 0.32, fontSize: 11, bold: true, color: WHITE, align: 'center', valign: 'middle'});
    txt(s, t, {x: x + 0.48, y: y + i * gap, w: w - 0.48, h: gap - 0.05, fontSize: size, color});
  });
}

// 1 · Title
{
  const s = pres.addSlide({masterName: 'DARK'});
  txt(s, 'ROBT613 · MIDTERM PROJECT', {x: 0.8, y: 1.0, w: 8, h: 0.35, fontSize: 13, bold: true, color: CORAL, charSpacing: 2});
  txt(s, 'Can a P300 decoder skip calibration?', {x: 0.8, y: 1.5, w: 8.3, h: 1.7, fontFace: HEAD, fontSize: 44, bold: true, color: WHITE});
  txt(s, 'Cross-subject transfer of a shrinkage-LDA ERP decoder on BrainInvaders2013a',
    {x: 0.8, y: 3.3, w: 7.8, h: 0.9, fontSize: 20, color: PALE});
  const names = ['Birzhan Zhunusbekov', 'Arman Abakov', 'Nazerke Khamzina', 'Bekmurat Amangeldiyev'];
  names.forEach((nm, i) => txt(s, nm, {x: 0.8 + (i % 2) * 3.6, y: 4.85 + Math.floor(i / 2) * 0.45, w: 3.4, h: 0.4, fontSize: 17, color: WHITE}));
  card(s, 9.6, 1.0, 3.1, 5.5, DEEP);
  [[String(n), 'subjects, IDs 8–24'], [String(2 * n), 'held-out test runs'], [`${m.setup_analysis_minutes.toFixed(1)} min`, 'fresh Colab CPU run']]
    .forEach(([v, l], i) => stat(s, 9.95, 1.35 + i * 1.7, 2.6, v, l, {color: CORAL, size: 40, labelColor: PALE}));
  const note = '45 s. Introduce the practical question: can a new user avoid labelled calibration? Make clear that this is an offline, retrospective analysis of BrainInvaders2013a, not a new online BCI. Introduce the four team members and their lead areas. Source: notebook introduction and results/run_manifest.json.';
  s.addNotes(note);
  notes.push({title: 'Can a P300 decoder skip calibration?', note});
}

// 2 · Question and hypotheses
{
  const s = slide('Can other users’ data replace calibration?', 'Research question',
    '60 s. Every P300 session normally starts with calibration. H1 asks whether training on other people beats chance. H2 compares transfer with subject-specific calibration. H3 matches training class counts so that quantity alone cannot explain a gap. Barachant and Congedo (2014) motivate generic initialization; Kim and Kim (2025) motivate practical zero-training use with methods outside our scope, which we do not implement. Sources: arXiv:1409.0107 and doi:10.1109/TNSRE.2025.3603979.');
  txt(s, 'Every P300 session normally starts with a calibration phase. If a decoder trained on other people works out of the box, a new user can start immediately.',
    {x: 0.6, y: 1.9, w: 4.5, h: 2.0, fontSize: 19});
  txt(s, 'We measure how much an in-scope linear decoder loses when it skips calibration, and for whom.',
    {x: 0.6, y: 4.05, w: 4.5, h: 1.3, fontSize: 16, color: SLATE});
  const H = [['H1', 'Transfer works', 'A leave-one-subject-out (LOSO) decoder ranks targets above chance (AUC > 0.5).'],
    ['H2', 'Calibration still helps', 'Within-subject training beats LOSO on the same held-out run.'],
    ['H3', 'Not just more data', 'Within still beats LOSO when training class counts are matched (LOSO-matched).']];
  H.forEach(([id, head, body], i) => {
    const y = 1.85 + i * 1.62;
    card(s, 5.6, y, 7.1, 1.42);
    s.addShape(pres.shapes.OVAL, {x: 5.85, y: y + 0.36, w: 0.7, h: 0.7, fill: {color: CORAL}, line: {color: CORAL}});
    txt(s, id, {x: 5.85, y: y + 0.36, w: 0.7, h: 0.7, fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, align: 'center', valign: 'middle'});
    txt(s, head, {x: 6.8, y: y + 0.2, w: 5.7, h: 0.4, fontSize: 19, bold: true});
    txt(s, body, {x: 6.8, y: y + 0.62, w: 5.7, h: 0.7, fontSize: 15, color: SLATE});
  });
  caption(s, 'Subjects 8–10 informed development in an earlier pilot; a sensitivity analysis excludes them (n = 14).');
}

// 3 · Data audit
{
  const s = slide('Data audit and provenance', 'Stage 1 · Data audit',
    '60 s. BrainInvaders2013a, subjects 8–24: each has exactly one session. Subjects 1–7 have eight sessions each (about 1 GB per subject), which does not fit the 30-minute Colab budget, so they are excluded by data size, not by performance. Every included run passes assertions for 80 targets, 400 non-targets, 16 EEG channels and 512 Hz. Units are volts in MNE, reported in µV. Data come from the original Zenodo files and are loaded through MOABB into MNE. Source: notebook stage 1; acquisition paper arXiv:1904.09111.',
    {stage: 0});
  const T = [[String(n), 'subjects (IDs 8–24), one session each'], ['16', 'EEG channels at 512 Hz, left-earlobe reference'],
    [String(2 * n), 'training runs in total, two per subject'], ['80 / 400', 'targets / non-targets per run (1 : 5)']];
  T.forEach(([v, l], i) => {
    const x = 0.6 + (i % 2) * 2.95, y = 1.9 + Math.floor(i / 2) * 1.8;
    card(s, x, y, 2.75, 1.6);
    stat(s, x + 0.25, y + 0.2, 2.3, v, l, {color: i === 3 ? CORAL : INK, size: 32, h: 1.3});
  });
  txt(s, 'Loaded from the original Zenodo files through MOABB into MNE Raw objects. Every run is checked by assertions before analysis.',
    {x: 0.6, y: 5.6, w: 5.6, h: 1.0, fontSize: 15, color: SLATE});
  image(s, 'cell_12_0.png', 6.6, 1.8, 3.0, 4.7);
  image(s, 'cell_13_1.png', 9.75, 1.8, 3.0, 4.7);
  caption(s, 'Left: sensor layout from the MOABB/MNE montage. Right: raw EEG excerpt with flash events.');
}

// 4 · Filtering
{
  const s = slide('Filter choice backed by the spectrum', 'Stage 2 · Signal inspection',
    '65 s. The PSD shows slow drift below 0.5 Hz and a sharp 50 Hz mains peak, strongest on P7. The FIR band-pass 0.5–20 Hz removes both: after the low-pass, 50 Hz is about −75 dB, so a notch adds nothing (measured in filter_measurements.csv). Data are resampled from 512 to 128 Hz together with the events. A 3.3 s edge guard (the filter half-support) keeps epochs away from recording edges. Filtering is done per run, so nothing crosses held-out runs. Zero-phase filtering uses future samples, so the analysis is offline. Source: notebook stage 2.',
    {stage: 1});
  const r = image(s, 'cell_18_0.png', 0.6, 1.75, 12.1, 3.45);
  const C = [['0.5–20 Hz', 'FIR band-pass, zero-phase, fitted per run'], ['128 Hz', 'resampled from 512 Hz together with the events'],
    ['No notch', '50 Hz is already about −75 dB after the low-pass'], ['3.3 s', 'edge guard equal to the filter half-support']];
  C.forEach(([v, l], i) => {
    const x = 0.6 + i * 3.07;
    card(s, x, 5.45, 2.9, 1.3);
    txt(s, v, {x: x + 0.22, y: 5.58, w: 2.5, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, color: i === 2 ? CORAL : INK});
    txt(s, l, {x: x + 0.22, y: 6.1, w: 2.5, h: 0.6, fontSize: 13, color: SLATE});
  });
  caption(s, 'Subject 8: PSD before and after band-pass (left) and the FIR frequency response (right).');
}

// 5 · Artifacts and QC
{
  const s = slide('Artifact handling and quality control', 'Stage 3 · Artifacts',
    `65 s. ICA (FastICA, 15 components) is fitted on a 1–20 Hz copy of each unlabelled run and the unmixing is applied to the main band-pass. Fp1/Fp2 serve as ocular proxies because there is no EOG channel. A component is removed only if both a proxy-correlation outlier rule and a frontal-map rule agree. The plot shows the blink-related deflection at Fp1 disappearing while the Pz response is preserved. Then a 100 µV peak-to-peak rejection runs on the cleaned epochs. Full-cohort counts: ${qc.edge_drop_T} target / ${qc.edge_drop_NT} non-target edge drops and ${qc.amp_drop_T} / ${qc.amp_drop_NT} amplitude drops; ${m.retained_percent.toFixed(2)}% of epochs retained. F6 was flagged as a bad channel and is ignored by rejection; it is not in the decoder montage. Residual artifacts remain possible. Source: notebook stage 3 and qc.csv.`,
    {stage: 2});
  image(s, 'cell_31_0.png', 0.6, 1.8, 8.0, 2.4, 'left');
  numbered(s, 0.6, 4.5, 8.0, [
    'FastICA, 15 components, fitted on a 1–20 Hz copy of each unlabelled run',
    'A component is removed only if the Fp1/Fp2 proxy correlation and the frontal map agree',
    'Peak-to-peak rejection at 100 µV on the cleaned epochs (threshold sweep in notebook)']);
  card(s, 9.1, 1.8, 3.6, 4.85);
  stat(s, 9.4, 2.0, 3.1, `${m.retained_percent.toFixed(1)}%`, 'of epochs retained across all 34 runs', {color: CORAL, size: 44, h: 1.6});
  txt(s, [
    {text: 'Edge drops', options: {bold: true, breakLine: true}},
    {text: `${qc.edge_drop_T} target / ${qc.edge_drop_NT} non-target`, options: {color: SLATE, breakLine: true}},
    {text: ' ', options: {fontSize: 8, breakLine: true}},
    {text: 'Amplitude drops', options: {bold: true, breakLine: true}},
    {text: `${qc.amp_drop_T} target / ${qc.amp_drop_NT} non-target`, options: {color: SLATE, breakLine: true}},
    {text: ' ', options: {fontSize: 8, breakLine: true}},
    {text: 'Bad channel', options: {bold: true, breakLine: true}},
    {text: `${m.observed_bad_channels.join(', ')}, outside the decoder montage`, options: {color: SLATE}},
  ], {x: 9.4, y: 3.75, w: 3.1, h: 2.8, fontSize: 15});
  caption(s, 'Subject 8, target − non-target at Fp1 and Pz before (dashed) and after ICA.');
}

// 6 · P300 check
{
  const s = slide('The P300 is there before we decode it', 'Stage 4 · Neurophysiology',
    `65 s. Before any classifier, check that the expected brain response exists. The Pz window 250–600 ms was fixed in advance (notebook 01; Vaineau et al. 2019 report 240–600 ms). Grey lines are individual subjects, red is the grand average; the topomaps show the centro-parietal positivity peaking around 350 ms. ${d.positive_erp} of ${n} subjects have a positive window mean, mean contrast ${d.mean_erp.toFixed(2)} µV. Single-epoch permutations are descriptive only because flashes overlap. The window is never chosen by decoding accuracy. Source: notebook stage 4 and erp.csv.`,
    {stage: 3});
  image(s, 'cell_46_0.png', 0.6, 1.75, 7.0, 3.35, 'left');
  image(s, 'cell_46_1.png', 0.6, 5.15, 7.0, 1.7, 'left');
  card(s, 8.1, 1.85, 4.6, 1.6);
  stat(s, 8.4, 2.0, 4.1, `${d.positive_erp} / ${n}`, 'subjects with a positive target − non-target mean at Pz', {color: CORAL, size: 36, h: 1.4});
  card(s, 8.1, 3.65, 4.6, 1.6);
  stat(s, 8.4, 3.8, 4.1, `${d.mean_erp.toFixed(2)} µV`, 'mean Pz contrast in the predeclared window', {size: 36, h: 1.4});
  txt(s, 'Window 250–600 ms fixed in advance (notebook 01). It is never tuned on decoding accuracy.',
    {x: 8.1, y: 5.55, w: 4.6, h: 1.0, fontSize: 15, color: SLATE});
}

// 7 · Features and classifier
{
  const s = slide('One leakage-free pipeline', 'Stage 5 · Features and classifier',
    '55 s. A feature is the mean voltage at one electrode during one 50 ms interval: five sites times sixteen windows over 0–800 ms gives 80 features. The montage was frozen before the full run and does not depend on held-out data. Everything that learns from data — the scaler and the analytic shrinkage of LDA — sits inside one scikit-learn Pipeline and is fitted on training folds only. No hyperparameters are tuned, so no nested CV is needed. The heatmap shows where the target − non-target difference lives for subject 8. Source: notebook stage 5.',
    {stage: 4});
  const B = [['Epoch', '0–800 ms after each flash, 128 Hz'], ['ERP features', '16 × 50 ms window means at Cz, P3, Pz, P4, Oz = 80'],
    ['StandardScaler', 'mean and SD from the training fold'], ['Shrinkage LDA', 'analytic (Ledoit–Wolf) shrinkage']];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x: 6.6, y: 1.72, w: 6.07, h: 1.88, fill: {color: 'FCEAE5'}, line: {color: CORAL, width: 1, dashType: 'dash'}, rectRadius: 0.1});
  txt(s, 'fitted on training folds only', {x: 6.8, y: 3.25, w: 5.75, h: 0.3, fontSize: 12, italic: true, color: CORAL, align: 'center'});
  B.forEach(([h, b], i) => {
    const x = 0.6 + i * 3.05;
    card(s, x, 1.85, 2.7, 1.35, i < 2 ? MIST : WHITE);
    txt(s, h, {x: x + 0.2, y: 1.97, w: 2.35, h: 0.4, fontSize: 17, bold: true});
    txt(s, b, {x: x + 0.2, y: 2.4, w: 2.35, h: 0.75, fontSize: 13, color: SLATE});
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, {x: x + 2.76, y: 2.37, w: 0.22, h: 0.3, fill: {color: SLATE}, line: {color: SLATE}});
  });
  image(s, 'cell_51_0.png', 0.6, 3.95, 6.0, 2.85, 'left');
  numbered(s, 7.0, 4.05, 5.7, [
    'One scikit-learn Pipeline, as in notebook 02',
    'No hyperparameters tuned, so nothing is chosen on test data',
    'Montage and windows frozen before the full-cohort run',
    'Same pipeline in all three conditions'], {gap: 0.68, size: 15});
}

// 8 · Evaluation design
{
  const s = slide('Same test epochs, different training data', 'Stage 6 · Honest evaluation',
    '65 s. All three conditions test exactly the same held-out run of each subject. Within trains on the subject’s other run (leave-one-run-out, never a random epoch split, because overlapping flashes share samples). LOSO trains on all runs of the other 16 users, and an assertion guarantees the test subject is never in training. LOSO-matched subsamples the other users to the same target and non-target counts as Within and averages ten seeded draws. Chance level comes from 200 label permutations within runs. Each subject’s score is the mean over its two test runs. Source: notebook stages 5–6 and comparison_runs.csv.',
    {stage: 5});
  const C = [['Within', COND.within, 'the subject’s own other run', 'Leave-one-run-out, 2 folds'],
    ['LOSO-matched', COND.loso_matched, 'other 16 users, subsampled to the Within class counts', '10 seeded draws averaged'],
    ['LOSO', COND.loso, 'all runs of the other 16 users', 'Test subject asserted absent']];
  C.forEach(([name, col, train, note], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.85, 3.85, 3.25);
    s.addShape(pres.shapes.OVAL, {x: x + 0.25, y: 2.08, w: 0.3, h: 0.3, fill: {color: col}, line: {color: col}});
    txt(s, name, {x: x + 0.7, y: 2.0, w: 3.0, h: 0.45, fontSize: 21, bold: true, color: col});
    txt(s, 'TRAINS ON', {x: x + 0.25, y: 2.65, w: 3.4, h: 0.25, fontSize: 11, bold: true, color: SLATE, charSpacing: 1});
    txt(s, train, {x: x + 0.25, y: 2.92, w: 3.4, h: 0.8, fontSize: 16});
    txt(s, 'TESTS ON', {x: x + 0.25, y: 3.75, w: 3.4, h: 0.25, fontSize: 11, bold: true, color: SLATE, charSpacing: 1});
    txt(s, 'the same held-out run', {x: x + 0.25, y: 4.02, w: 3.4, h: 0.4, fontSize: 16});
    txt(s, note, {x: x + 0.25, y: 4.55, w: 3.4, h: 0.4, fontSize: 13, italic: true, color: SLATE});
  });
  const K = [['Metric', 'ROC AUC (primary) and balanced accuracy; one score per subject'],
    ['Chance level', '200 label permutations within runs; AUC 0.5 baseline'],
    ['Variability', 'mean ± SD across subjects and across test folds']];
  K.forEach(([h, b], i) => {
    const x = 0.6 + i * 4.1;
    txt(s, h, {x, y: 5.4, w: 3.85, h: 0.35, fontSize: 15, bold: true, color: CORAL});
    txt(s, b, {x, y: 5.78, w: 3.85, h: 0.8, fontSize: 14, color: SLATE});
  });
}

// 9 · Mean AUC
{
  const s = slide('Transfer works, calibration still wins', 'Results · Mean AUC',
    `65 s. Bars are means over ${n} subjects; each subject is the mean of its two held-out runs, so the SD is between-subject spread. Within reaches ${f3(d.means.within)}, LOSO ${f3(d.means.loso)}, LOSO-matched ${f3(d.means.loso_matched)}. LOSO-matched is lower than full LOSO, so the amount of external data matters. All 34 LOSO test runs beat their permutation null at p < .05. No bar comes from a random epoch split. Source: comparison_subjects.csv.`,
    {stage: 5});
  const keys = ['within', 'loso_matched', 'loso'], labels = ['Within', 'LOSO-matched', 'LOSO'];
  s.addChart(pres.charts.BAR, [{name: 'Mean ROC AUC × 1000', labels, values: keys.map(k => Math.round(d.means[k] * 1000))}], {
    x: 0.6, y: 1.75, w: 7.4, h: 4.95, barDir: 'col', barGapWidthPct: 70,
    chartColors: keys.map(k => COND[k]), showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: AUC_FMT,
    dataLabelFontSize: 16, dataLabelFontBold: true, dataLabelColor: INK, dataLabelFontFace: BODY,
    valAxisMinVal: 0, valAxisMaxVal: 1000, valAxisMajorUnit: 250, valAxisLabelFormatCode: AUC_FMT,
    valAxisLabelFontSize: 13, catAxisLabelFontSize: 15, valAxisLabelColor: SLATE, catAxisLabelColor: INK,
    valAxisLabelFontFace: BODY, catAxisLabelFontFace: BODY,
    valGridLine: {color: LINE, size: 0.75}, catGridLine: {style: 'none'}, catAxisLineShow: false, valAxisLineShow: false,
    showLegend: false, showValAxisTitle: true, valAxisTitle: 'ROC AUC', valAxisTitleFontSize: 13, valAxisTitleColor: SLATE,
  });
  keys.forEach((k, i) => {
    const y = 1.9 + i * 1.18;
    s.addShape(pres.shapes.OVAL, {x: 8.55, y: y + 0.08, w: 0.26, h: 0.26, fill: {color: COND[k]}, line: {color: COND[k]}});
    txt(s, labels[i], {x: 8.95, y, w: 3.6, h: 0.4, fontSize: 16, bold: true, color: COND[k]});
    txt(s, `${f3(d.means[k])} ± ${f3(d.sds[k])}`, {x: 8.95, y: y + 0.4, w: 3.8, h: 0.55, fontFace: HEAD, fontSize: 26, bold: true});
  });
  card(s, 8.55, 5.5, 4.15, 1.2, 'FCEAE5');
  txt(s, '34 / 34 LOSO test runs beat their permutation null (p < .05)', {x: 8.8, y: 5.62, w: 3.7, h: 1.0, fontSize: 15, bold: true, color: CORAL, valign: 'middle'});
  caption(s, `Mean ± SD across ${n} subjects; each subject = mean of 2 held-out runs. Chance AUC = 0.5.`);
}

// 10 · Individual subjects
{
  const s = slide('Every user transfers, but not equally', 'Results · Per subject',
    `65 s. Each grey line is one person (mean of two test runs); the coral line is the cohort mean and the slate line is chance. ${d.subjects.filter(r => r.within > r.loso).length} of ${n} users lose AUC without calibration. The median Within − LOSO gap (${f3(d.median_gap)}) is several times the median Within fold SD (${f3(d.median_fold_sd)}), though two folds estimate variability poorly. The weakest LOSO subject still reaches ${f3(Math.min(...d.subjects.map(r => r.loso)))}. Source: comparison_subjects.csv.`,
    {stage: 5});
  const labels = ['Within', 'LOSO-matched', 'LOSO'];
  const series = d.subjects.map(r => ({name: `S${r.subject}`, labels, values: [r.within, r.loso_matched, r.loso].map(v => Math.round(v * 1000))}));
  series.push({name: 'Mean', labels, values: [d.means.within, d.means.loso_matched, d.means.loso].map(v => Math.round(v * 1000))});
  series.push({name: 'Chance', labels, values: [500, 500, 500]});
  s.addChart(pres.charts.LINE, series, {
    x: 0.6, y: 1.75, w: 8.4, h: 4.95,
    chartColors: [...d.subjects.map(() => 'B8C2D1'), CORAL, SLATE], lineSize: 2, lineDataSymbol: 'none',
    valAxisMinVal: 400, valAxisMaxVal: 1000, valAxisMajorUnit: 100, valAxisLabelFormatCode: AUC_FMT,
    valAxisLabelFontSize: 13, catAxisLabelFontSize: 15, valAxisLabelColor: SLATE, catAxisLabelColor: INK,
    valAxisLabelFontFace: BODY, catAxisLabelFontFace: BODY,
    valGridLine: {color: LINE, size: 0.75}, catGridLine: {style: 'none'}, showLegend: false,
    showValAxisTitle: true, valAxisTitle: 'ROC AUC', valAxisTitleFontSize: 13, valAxisTitleColor: SLATE,
  });
  const S = [[`${d.subjects.filter(r => r.within > r.loso).length} / ${n}`, 'users score higher with their own calibration', CORAL],
    [f3(d.median_gap), 'median Within − LOSO gap', INK],
    [f3(d.median_fold_sd), 'median Within fold SD (2 folds)', INK],
    [f3(Math.min(...d.subjects.map(r => r.loso))), 'lowest LOSO AUC, still above chance', INK]];
  S.forEach(([v, l, c], i) => stat(s, 9.5, 1.85 + i * 1.22, 3.3, v, l, {color: c, size: 28, h: 1.1}));
  caption(s, 'Grey: individual subjects. Coral: cohort mean. Slate: chance (0.5).');
}

// 11 · Hypothesis tests
{
  const s = slide('All three hypotheses hold in this cohort', 'Stage 7 · Hypothesis test',
    `65 s. One subject is one unit. The three directional comparisons use paired Wilcoxon signed-rank tests with Holm correction. Read the median difference together with the adjusted p-value. Excluding pilot subjects 8–10 (n = 14) gives the same conclusions. These are conventional cohort summaries: LOSO models share training users, so p-values are not exact guarantees for new populations. Source: hypothesis_tests.csv and sensitivity_tests.csv.`,
    {stage: 6});
  const lab = ['LOSO > chance', 'Within > LOSO', 'Within > LOSO-matched'];
  d.tests.forEach((r, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.85, 3.85, 3.4);
    s.addShape(pres.shapes.OVAL, {x: x + 0.25, y: 2.08, w: 0.62, h: 0.62, fill: {color: CORAL}, line: {color: CORAL}});
    txt(s, `H${i + 1}`, {x: x + 0.25, y: 2.08, w: 0.62, h: 0.62, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, align: 'center', valign: 'middle'});
    txt(s, lab[i], {x: x + 1.05, y: 2.15, w: 2.7, h: 0.5, fontSize: 17, bold: true, valign: 'middle'});
    txt(s, `+${f3(r.median_diff)}`, {x: x + 0.25, y: 2.95, w: 3.4, h: 0.75, fontFace: HEAD, fontSize: 40, bold: true, color: INK});
    txt(s, 'median AUC difference', {x: x + 0.25, y: 3.7, w: 3.4, h: 0.3, fontSize: 13, color: SLATE});
    txt(s, [{text: 'Holm p = '}, ...sci(r.p_holm_primary)], {x: x + 0.25, y: 4.15, w: 3.4, h: 0.4, fontSize: 17, bold: true});
    txt(s, `rank-biserial r = ${r.rank_biserial.toFixed(2)}`, {x: x + 0.25, y: 4.6, w: 3.4, h: 0.35, fontSize: 13, color: SLATE});
  });
  card(s, 0.6, 5.5, 12.1, 1.15, 'FCEAE5');
  txt(s, [
    {text: 'Sensitivity without pilot subjects 8–10 (n = 14): ', options: {bold: true, color: CORAL}},
    {text: 'Holm p = '}, ...sci(d.sensitivity[0].p_holm_primary), {text: ' / '}, ...sci(d.sensitivity[1].p_holm_primary),
    {text: ' / '}, ...sci(d.sensitivity[2].p_holm_primary), {text: ' for H1 / H2 / H3. Same conclusions.'},
  ], {x: 0.85, y: 5.6, w: 11.6, h: 0.95, fontSize: 15, valign: 'middle'});
  caption(s, `Paired Wilcoxon signed-rank tests across ${n} subjects, Holm-corrected for the three primary hypotheses.`);
}

// 12 · Conclusion
{
  const s = slide('What the result shows', 'Conclusion',
    '45 s. State the outcome as conditional cohort evidence: transfer works well above chance, but calibration still adds about 0.07 AUC, and the amount of external data matters. Then stress the limits: offline preprocessing, flash-level AUC instead of command accuracy, one acquisition protocol, and dependent LOSO models. A negative result would have been equally valid under the rubric. Every teammate should be ready to explain any stage. Source: notebook conclusions and run_manifest.json.',
    {dark: true});
  txt(s, [
    {text: 'A decoder trained only on other users works on a new user ', options: {color: WHITE}},
    {text: `(AUC ${f3(d.means.loso)})`, options: {color: CORAL}},
    {text: ', but personal calibration still adds ', options: {color: WHITE}},
    {text: `${f3(d.median_gap)}`, options: {color: CORAL}},
    {text: ' AUC (median).', options: {color: WHITE}},
  ], {x: 0.6, y: 1.9, w: 6.6, h: 2.6, fontFace: HEAD, fontSize: 30, bold: true});
  txt(s, 'More external data helps: matching the training size lowers transfer AUC from 0.810 to 0.724.',
    {x: 0.6, y: 4.7, w: 6.4, h: 1.2, fontSize: 17, color: PALE});
  card(s, 7.8, 1.85, 4.9, 4.8, DEEP);
  txt(s, 'LIMITS', {x: 8.15, y: 2.1, w: 4.2, h: 0.3, fontSize: 12, bold: true, color: CORAL, charSpacing: 2});
  numbered(s, 8.15, 2.6, 4.3, [
    'Offline: zero-phase filters and per-run ICA use future samples',
    'Single-flash AUC is not online command accuracy',
    `One protocol, ${n} single-session users`,
    'LOSO models share training users; p-values summarise this cohort'], {color: WHITE, size: 15, gap: 0.95});
}

// 13 · Backup: references and AI disclosure
{
  const s = slide('References and AI disclosure', 'Backup',
    'Backup slide. References are background and motivation, not sources of results. The core pipeline follows workshop notebooks 00–06 and cites them where code is reused. AI tools: OpenAI Codex for code review, debugging, API lookup, implementation, statistics checks, plotting and text drafting; Claude Code for checking the submission against the assignment and editing wording. Every number comes from the executed notebook.');
  numbered(s, 0.6, 1.9, 7.0, [
    'E. Vaineau et al., “Brain Invaders adaptive versus non-adaptive P300 BCI dataset,” arXiv:1904.09111, 2019',
    'A. Barachant, M. Congedo, “A Plug&Play P300 BCI using information geometry,” arXiv:1409.0107, 2014',
    'J. Kim, S.-P. Kim, “A Plug-and-Play P300-based BCI with zero-training application,” IEEE TNSRE, vol. 33, 2025',
    'V. Jayaram, A. Barachant, “MOABB: trustworthy algorithm benchmarking for BCIs,” J. Neural Eng., 2018',
    'B. Abibullaev, “ROBT613 BCI Workshop,” notebooks 00–06, github.com/berdakh/ROBT613-BCI-Workshop'],
    {fill: INK, size: 14, gap: 0.92});
  card(s, 8.1, 1.85, 4.6, 4.8);
  txt(s, 'AI TOOLS', {x: 8.4, y: 2.1, w: 4.0, h: 0.3, fontSize: 12, bold: true, color: CORAL, charSpacing: 2});
  txt(s, [
    {text: 'OpenAI Codex', options: {bold: true, breakLine: true}},
    {text: 'code review, debugging, API lookup, implementation, statistics checks, plotting, text drafting', options: {color: SLATE, breakLine: true}},
    {text: ' ', options: {fontSize: 8, breakLine: true}},
    {text: 'Claude Code', options: {bold: true, breakLine: true}},
    {text: 'requirements check, wording edits, slide generator', options: {color: SLATE, breakLine: true}},
    {text: ' ', options: {fontSize: 8, breakLine: true}},
    {text: 'No AI tool generated results, citations or dataset facts. Every number comes from the executed notebook.', options: {italic: true}},
  ], {x: 8.4, y: 2.55, w: 4.0, h: 4.0, fontSize: 15});
}

// 14 · Backup: reproducibility
{
  const s = slide('Reproducibility record', 'Backup',
    `Backup slide. Direct dependencies are pinned in requirements.txt and actual versions are recorded in the run manifest. The verified run used a fresh Google Colab CPU session with an empty data cache: ${m.setup_analysis_minutes.toFixed(2)} minutes including downloads, all 32 code cells, no errors. The first install needs one runtime restart to reload NumPy. The downloader respects Zenodo rate limits. Source: results/run_manifest.json and results/colab_verification.json.`);
  const rows = [[{text: 'Package', options: {bold: true, color: WHITE, fill: {color: INK}}}, {text: 'Version', options: {bold: true, color: WHITE, fill: {color: INK}}}],
    ...Object.entries(m.versions).map(([k, v], i) => [k, v].map(t => ({text: t, options: {fill: {color: i % 2 ? WHITE : MIST}}})))];
  s.addTable(rows, {x: 0.6, y: 1.85, w: 5.0, colW: [3.0, 2.0], fontFace: BODY, fontSize: 13, color: INK, rowH: 0.4,
    border: {type: 'solid', pt: 0.5, color: LINE}, margin: [0.04, 0.12, 0.04, 0.12]});
  const T = [[`${m.setup_analysis_minutes.toFixed(2)} min`, 'fresh Colab CPU run with an empty data cache'], [String(m.seed), 'random seed for ICA, permutations and draws'],
    [String(n), 'subjects, IDs 8–24 listed in the manifest'], ['0', 'errors across all 32 code cells']];
  T.forEach(([v, l], i) => {
    const x = 6.3 + (i % 2) * 3.25, y = 1.85 + Math.floor(i / 2) * 2.1;
    card(s, x, y, 3.05, 1.85);
    stat(s, x + 0.25, y + 0.25, 2.6, v, l, {color: i === 0 ? CORAL : INK, size: 32, h: 1.4});
  });
  txt(s, 'Shared code lives in functions; the notebook runs top to bottom with versions pinned.',
    {x: 6.3, y: 6.15, w: 6.4, h: 0.5, fontSize: 14, color: SLATE});
}

await pres.writeFile({fileName: path.join(ROOT, 'output/presentation.pptx')});
fs.writeFileSync(path.join(ROOT, 'output/SPEAKER_NOTES.md'),
  '# Speaker notes\n\n' + notes.map((r, i) => `## ${i + 1}. ${r.title}\n\n${r.note}\n`).join('\n'));
console.log(`Wrote output/presentation.pptx (${notes.length} slides) and output/SPEAKER_NOTES.md`);
