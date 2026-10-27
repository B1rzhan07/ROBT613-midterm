import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const ink='#182D42', blue='#216A93', muted='#506477', paper='#FFFFFF';
export async function buildPresentation(p){
 const d=JSON.parse(await fs.readFile(path.join(ROOT,'results/presentation_data.json'),'utf8'));
 const m=d.manifest,n=m.n_subjects,t=m.qc_totals;
 const chartNumber=x=>Number(x.toFixed(12));
 const fmt=x=>Number(x).toFixed(3), pv=x=>x<.001?Number(x).toExponential(2):Number(x).toFixed(4);
 const notes=[];
 p.theme.defaultFont='Arial';
 for (const old of p.slides.items) old.delete();
 function text(s,value,x,y,w,h,size=30,color=ink,bold=false){
  const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=value;a.text.style={typeface:'Arial',fontSize:size,color,bold};return a;
 }
 function slide(title,note){
  const s=p.slides.add();s.background.fill=paper;
  text(s,title,60,42,1160,90,43,ink,true);
  text(s,String(p.slides.items.length),1160,671,55,25,17,muted);
  s.speakerNotes.text=note;notes.push({title,note});return s;
 }
 function table(s,values,x=60,y=160,w=1160,h=350,widths){
  const tab=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
  tab.borders.assign({fill:'#D6E0E7',width:1,style:'solid'});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
   const cell=tab.getCell(r,c);cell.fill=r===0?ink:'#FFFFFF';
   cell.text.style={typeface:'Arial',fontSize:25,color:r===0?'#FFFFFF':ink,bold:r===0};
  }
  return tab;
 }
 async function image(s,cell,x,y,w,h,variant=-1){
  const names=(await fs.readdir(path.join(ROOT,'results/figures'))).filter(n=>n.startsWith(`cell_${String(cell).padStart(2,'0')}_`)).sort();
  const name=names.at(variant);if(!name)throw Error('Missing figure '+cell);
  const b=await fs.readFile(path.join(ROOT,'results/figures',name));
  s.images.add({path:path.join(ROOT,'results/figures',name),alt:'Executed EEG notebook evidence',fit:'contain',position:{left:x,top:y,width:w,height:h}});
 }
 let s=slide('Calibration-free P300 decoding','45 s. Introduce the practical question: can a new user avoid labelled calibration? Explain that this is an offline retrospective analysis of BrainInvaders2013a, not a new online BCI. Names supplied by the team. Source: notebook introduction and results/run_manifest.json.');
 s.shapes.items[0].text.fontSize=56;
 text(s,'Cross-subject transfer on BrainInvaders2013a',60,200,1080,110,40,blue);
 text(s,'Birzhan Zhunusbekov     Arman Abakov\nNazerke Khamzina          Bekmurat Amangeldiyev',60,380,1100,130,28);
 text(s,`${n} subjects   •   Linear ERP decoder   •   Offline evaluation`.replaceAll('   •   ','    '),60,575,1100,45,25,muted);
 s=slide('Research question and hypotheses','60 s. H1 asks whether external-subject training beats chance. H2 compares transfer with subject-specific calibration. H3 matches training class counts so quantity alone cannot explain a gap. The 2014 information-geometry paper motivates generic initialization and adaptation. Kim and Kim 2025 motivates practical zero-training use, using out-of-scope methods that we do not implement. Sources: https://arxiv.org/abs/1409.0107 and https://doi.org/10.1109/TNSRE.2025.3603979 (abstract consulted).');
 text(s,'Can other users’ data replace a new user’s calibration?',60,155,1140,100,37,blue,true);
 text(s,'H1   LOSO ranks targets above chance\n\nH2   Within-subject training exceeds LOSO\n\nH3   The gap remains with matched training counts',60,290,1140,290,30);
 text(s,'Earlier subjects 8–10 informed development. We disclose that boundary.',60,610,1140,40,23,muted);
 s=slide('Data and provenance','60 s. Subjects 8-24 each have one session. Explain why subjects 1-7 are excluded: their longitudinal archives are substantially larger. Every included run must pass assertions for 80 targets, 400 non-targets, 16 EEG channels and 512 Hz. File metadata determines adaptive condition. All data still load through MOABB into MNE. Source: notebook stage 1; acquisition paper https://arxiv.org/abs/1904.09111.');
 table(s,[['Property','Audited value'],['Subjects','8–24, n = 17'],['Runs','2 training runs per subject'],['Acquisition','16 EEG channels, 512 Hz'],['Reference','Left earlobe'],['Events','80 targets / 400 non-targets per run']],60,155,650,405,[250,400]);
 await image(s,12,750,155,450,405);
 text(s,'Original Zenodo training files. MOABB creates the MNE objects.',60,605,1150,50,24,muted);
 s=slide('Filtering and resampling','65 s. Explain evidence for 0.5-20 Hz rather than presenting defaults. Slow drift motivates high-pass filtering. The low-pass suppresses mains activity, assessed in residual measurements. Events resample together with the data. A half-support edge guard keeps epochs away from recording boundaries. No filtering crosses held-out runs. Zero phase uses future samples, so the conclusion is offline. Source: notebook stage 2 and filter_measurements.csv.');
 await image(s,18,50,150,800,430);
 text(s,'FIR 0.5–20 Hz\n\n512 to 128 Hz\n\nEvents resampled together\n\nRecording-edge guard',900,170,320,390,27);
 text(s,'Per-run processing preserves run boundaries',60,605,1150,45,27,blue,true);
 s=slide('Artifact handling and quality control','65 s. ICA fits a 1-20 Hz copy of each unlabelled run and applies the unmixing to the main band-pass. Fp1/Fp2 are ocular proxies, not measured EOG. Exclusion requires both a proxy correlation outlier and a frontal map. The example plot is not a claim that every ocular artifact disappears. Present the full-cohort retention counts at right. A flagged core site or excessive bad channels stops the notebook for review. Source: notebook stage 3 and qc.csv.');
 await image(s,31,50,150,790,420);
 text(s,`${m.retained_percent.toFixed(2)}% retained`,880,170,330,70,36,blue,true);
 text(s,`Edge drops\n${t.edge_drop_T} target / ${t.edge_drop_NT} non-target\n\nAmplitude drops\n${t.amp_drop_T} target / ${t.amp_drop_NT} non-target\n\n100 µV peak-to-peak`,880,270,330,310,25);
 text(s,'ICA is offline and label-free. Residual artifacts remain possible.',60,610,1150,45,24,muted);
 s=slide('P300 physiology before decoding','65 s. Establish a brain-signal check before classification. The preselected Pz window is 250-600 ms. Grand-average plots weight subjects equally. Point to the positive centro-parietal response and the individual variation. Single-epoch permutations are descriptive because flashes overlap. The physiological check never selects decoder features by their test accuracy. Source: notebook stage 4 and erp.csv.');
 await image(s,46,50,145,840,295,0);
 await image(s,46,50,445,840,170);
 text(s,'Fixed Pz window\n250–600 ms',940,170,280,120,29,blue,true);
 text(s,`${d.positive_erp}/${n} positive\nsubject window means\n\nMean contrast\n${d.mean_erp.toFixed(2)} µV`,940,330,280,220,27);
 s=slide('Features and linear classifier','55 s. Describe a feature as the mean voltage at one electrode during one 50 ms interval. Five sites times sixteen windows gives eighty features. The montage was frozen before the full run and does not depend on held-out flags. The earlier three-person pilot informed the revision. Scaling and shrinkage fit training data only. No hyperparameters are tuned against test AUC. Source: notebook stage 5.');
 table(s,[['Step','Fixed specification'],['Channels','Cz, P3, Pz, P4, Oz'],['ERP features','16 × 50 ms windows over 0–800 ms'],['Dimension','5 × 16 = 80 features'],['Scaling','Training-fold StandardScaler'],['Classifier','Shrinkage LDA, analytic shrinkage']],60,155,1160,390,[300,860]);
 text(s,'No pooled bad-channel flags shape training features',60,605,1150,45,28,blue,true);
 s=slide('Evaluation on the same held-out epochs','65 s. All rows test exactly the same epochs for each subject/run. Within uses the other run from that person. LOSO uses all other people, with an assert excluding the test subject. Matched transfer preserves both target and non-target training counts and averages ten seeded draws. AUC ranks flashes and is distinct from complete command accuracy. Source: notebook stages 5-6 and comparison_runs.csv.');
 table(s,[['Condition','Training data','Test data'],['Within','Subject’s other run','Same held-out run'],['LOSO-matched','Other users, matched class counts','Same held-out run'],['LOSO','All runs of the other 16 users','Same held-out run']],60,160,1160,310,[245,600,315]);
 text(s,'Run groups for Within. Entire subjects held out for transfer.\nChance AUC = 0.5. One mean over two test runs per subject.',60,530,1150,110,28);
 s=slide('Mean AUC across subjects','65 s. State that the native chart uses subject means, and the printed SD is between-subject spread rather than fold spread. No bar represents a single random split. Interpret the observed ordering without assuming the hypothesis must succeed. Source: comparison_subjects.csv; exact values also appear in the report.');
 s.charts.add('bar',{position:{left:60,top:150,width:840,height:450},categories:['Within','LOSO-matched','LOSO'],series:[{name:'Mean ROC AUC',values:[d.means.within,d.means.loso_matched,d.means.loso].map(chartNumber),fill:blue}],barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:false,yAxis:{min:0,max:1,majorUnit:.2,title:{text:'ROC AUC',textStyle:{typeface:'Arial',fontSize:22}},textStyle:{typeface:'Arial',fontSize:21},majorGridlines:{fill:'#D6E0E7',width:1}},xAxis:{textStyle:{typeface:'Arial',fontSize:24}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{typeface:'Arial',fontSize:25}}});
 text(s,'Mean ± SD',940,165,280,55,27,muted,true);
 text(s,`Within\n${fmt(d.means.within)} ± ${fmt(d.sds.within)}\n\nMatched\n${fmt(d.means.loso_matched)} ± ${fmt(d.sds.loso_matched)}\n\nLOSO\n${fmt(d.means.loso)} ± ${fmt(d.sds.loso)}`,940,240,280,360,28);
 s=slide('Individual transfer performance','65 s. Each thin line is one person, averaged across two test runs. The thick line is the cohort mean, and the dotted line is chance. Highlight variation rather than promising universal performance. Compare the median paired gap with the two-fold spread, while noting that two folds estimate variability poorly. Source: comparison_subjects.csv.');
 const series=d.subjects.map(r=>({name:`Subject ${r.subject}`,values:[r.within,r.loso_matched,r.loso].map(chartNumber),line:{fill:'#AABEC9',width:1.5}}));
 series.push({name:'Mean',values:[d.means.within,d.means.loso_matched,d.means.loso].map(chartNumber),line:{fill:blue,width:5}});
 series.push({name:'Chance',values:[.5,.5,.5],line:{fill:'#707070',width:2,style:'dashed'}});
 s.charts.add('line',{position:{left:60,top:150,width:1140,height:420},categories:['Within','LOSO-matched','LOSO'],series,lineOptions:{smooth:false},hasLegend:false,yAxis:{min:.4,max:1,majorUnit:.1,numberFormatCode:'0.0',textStyle:{typeface:'Arial',fontSize:22},majorGridlines:{fill:'#E0E7EB',width:1}},xAxis:{textStyle:{typeface:'Arial',fontSize:25}}});
 text(s,`Median Within–LOSO gap: ${fmt(d.median_gap)}    Median Within fold SD: ${fmt(d.median_fold_sd)}`,60,610,1150,45,27);
 s=slide('Paired hypothesis tests','65 s. One subject is one unit. The three directional primary comparisons use Wilcoxon signed-rank tests and Holm correction. Read both the median difference and adjusted p-value. Pilot-excluded sensitivity uses fourteen evaluation subjects. The tests are conventional cohort summaries: LOSO models share training users, and p-values are not exact guarantees for new populations. Source: hypothesis_tests.csv and sensitivity_tests.csv.');
 table(s,[['Hypothesis','Median gap','Holm p'],...d.tests.map(r=>[r.hypothesis.replace('H1: ','H1 ').replace('H2: ','H2 ').replace('H3: ','H3 '),fmt(r.median_diff),pv(r.p_holm_primary)])],60,160,1160,300,[650,250,260]);
 text(s,`Pilot-excluded sensitivity: n = 14\nHolm p for H1 / H2 / H3: ${d.sensitivity.map(r=>pv(r.p_holm_primary)).join(' / ')}`,60,525,1160,100,27);
 s=slide('Conclusion and limits','45 s. State the outcome as conditional cohort evidence. Stress the limits of flash AUC, offline preprocessing and one acquisition protocol. A different outcome would still be valid under the project rubric. Explain the verified Colab CPU timing in the backup record. Every teammate should be prepared to explain all stages. Source: notebook conclusions and run_manifest.json.');
 const supported=d.tests.filter(r=>r.p_holm_primary<.05).map(r=>r.hypothesis.slice(0,2)).join(', ')||'None';
 text(s,`Supported in this cohort: ${supported}`,60,160,1150,90,37,blue,true);
 text(s,'Transfer performance must be assessed per user\n\nOffline cleaning uses future unlabelled samples\n\nSingle-flash AUC does not establish online command accuracy\n\nIndependent cohorts are needed for stronger generalization claims',60,285,1150,310,29);
 s=slide('References and AI disclosure','Backup slide. Acquisition and motivation references are background, not imported results. The core follows workshop notebooks 00-06. Codex assisted with implementation, execution, API lookup, statistics, plotting and drafting in this session. All numbers come from the executed notebook. Earlier AI usage remains for the team to confirm. Authors must review and be able to explain every line.');
 text(s,'Vaineau et al. (2019). Brain Invaders dataset. arXiv:1904.09111\n\nBarachant & Congedo (2014). Plug&Play P300. arXiv:1409.0107\n\nKim & Kim (2025). Zero-training P300. DOI: 10.1109/TNSRE.2025.3603979\n\nROBT613 BCI Workshop, notebooks 00–06',60,155,1150,310,25);
 text(s,'AI assistance: OpenAI Codex supported implementation, execution,\nchecks and drafting. The team reviews and defends the submitted work.',60,535,1150,100,26,muted);
 s=slide('Reproducibility record',`Backup slide. Exact direct dependencies are pinned and actual versions are in the manifest. The notebook saves figures and CSVs. Execution environment: ${m.environment}. Cold cache at setup: ${m.cold_cache_at_start ?? false}. Initial installation required one restart to reload NumPy. The downloader respects Zenodo rate limits. Source: results/run_manifest.json. Saved notebook: https://colab.research.google.com/drive/19qYoD5urhaAcdMhhokzYAZPZuN2ra2Rh`);
 table(s,[['Package','Version'],...Object.entries(m.versions)],60,155,650,430,[360,290]);
 text(s,`${n} subjects\nSeed ${m.seed}\n\nFull run: ${m.setup_analysis_minutes.toFixed(2)} min\n${m.environment}\n\nEmpty data cache\n${m.cold_cache_at_start?'Verified':'Not measured'}`,780,170,430,380,29);
 await fs.mkdir(path.join(ROOT,'output/pdf'),{recursive:true});await fs.mkdir(path.join(ROOT,'.build/slides'),{recursive:true});
 await fs.writeFile(path.join(ROOT,'output/SPEAKER_NOTES.md'),notes.map((r,i)=>`## ${i+1}. ${r.title}\n\n${r.note}\n`).join('\n'));
 return {slides:p.slides.items.length};
}
