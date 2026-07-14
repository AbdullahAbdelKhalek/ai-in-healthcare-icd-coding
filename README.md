# Automated ICD-9 Coding from Clinical Notes

Training a model to read a hospital **discharge summary** and assign its **ICD-9 diagnosis codes**, the labels a
medical coder assigns for billing, quality reporting, and research. The project builds the idea up in two stages:
a transparent linear baseline, and then a convolutional neural network with **per-label attention** that, for each
predicted code, highlights the phrase in the note it attended to.

This README is written to stand on its own. It explains the clinical motivation, the data, every modeling choice,
the results, and how to reproduce the work, so you do not need the notebook or slides to understand what was done.

- **Task:** multi-label text classification (a note can carry many diagnosis codes at once).
- **Data:** MIMIC-III (credentialed), tables `NOTEEVENTS`, `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES`.
- **Models:** TF-IDF + logistic regression, then a CNN with per-label attention (the CAML architecture).
- **Compute:** runs end to end on a standard CPU in about 20 minutes.

## Contents

1. [Background](#background)
2. [The task](#the-task)
3. [The data](#the-data)
4. [Building the cohort](#building-the-cohort)
5. [Preprocessing](#preprocessing)
6. [How success is measured](#how-success-is-measured)
7. [The baseline model](#the-baseline-model)
8. [The deep model](#the-deep-model)
9. [Results](#results)
10. [Reading the model's evidence](#reading-the-models-evidence)
11. [Repository layout](#repository-layout)
12. [Running it yourself](#running-it-yourself)
13. [Reproducibility](#reproducibility)
14. [Limitations](#limitations)
15. [Data use and privacy](#data-use-and-privacy)
16. [Reference](#reference)

## Background

When a patient is discharged, a clinician writes a free-text discharge summary describing the history, hospital
course, results, and diagnoses. A certified medical coder then reads that narrative and assigns ICD codes, a
controlled vocabulary in which every diagnosis has a standardized identifier. For example, `4019` is essential
hypertension and `4280` is congestive heart failure.

These codes are not administrative trivia. They determine hospital reimbursement, they feed quality and safety
reporting, and researchers use them to define patient cohorts. Manual coding is labor-intensive, and trained
coders disagree with one another more often than one might expect. That combination, a narrow, high-volume,
text-in and labels-out task, is a reasonable place to apply machine learning as a first-pass suggestion tool that
a human confirms.

## The task

Predicting the codes for one note is **multi-label classification**, not multi-class. In multi-class problems the
model picks exactly one category. Here a single admission usually carries several diagnoses at once, so the
correct answer is a subset of codes and the model must decide yes or no for each code independently.

Following a top-50 MIMIC-III benchmark-style setup, the project uses the 50 most frequent ICD-9 codes. This keeps
the problem tractable on a laptop while still covering the diagnoses that appear constantly in an intensive-care
population: hypertension, heart failure, atrial fibrillation, acute kidney failure, diabetes, and so on. The full
problem has thousands of codes and is considerably harder.

- **Input:** the text of one discharge summary.
- **Output:** which of the 50 codes apply (a 50-way independent yes/no).

## The data

The project uses **MIMIC-III**, a large, publicly available but credentialed database of de-identified
intensive-care records. Three tables are used:

| Table | Role |
|---|---|
| `NOTEEVENTS` | the clinical notes; only discharge summaries are kept |
| `DIAGNOSES_ICD` | the ICD-9 codes assigned to each admission (the labels) |
| `D_ICD_DIAGNOSES` | a readable title for each code |

MIMIC-III is not distributed here. You must obtain it yourself from PhysioNet (free credentialing is required) and
place the three CSV files in a `data/` folder, as described under [Running it yourself](#running-it-yourself).

## Building the cohort

Three construction choices matter, because they are where this kind of project usually goes wrong.

- **One document per admission.** A discharge summary can arrive as a main report plus later addenda. These are
  concatenated so each admission is a single document.
- **Top-50 labels.** The 50 most frequent codes among admissions that have a discharge summary are selected, and
  the cohort is limited to admissions that carry at least one of them.
- **Split by patient, never by note.** If the same patient appeared in both the training and test sets, the model
  could memorize patient-specific wording and look better than it really is. Splitting on patient id removes that
  leakage.

The resulting cohort:

| Quantity | Value |
|---|---|
| Admissions (discharge summaries) | 16,410 |
| Unique patients | 12,687 |
| Diagnosis codes (labels) | 50 |
| Average codes per note | 4.65 |
| Median note length | ~1,367 words |
| Train / validation / test | 13,075 / 1,732 / 1,603 admissions |

## Preprocessing

The text cleaning is deliberately simple, so it can be stated in one sentence and defended.

- MIMIC de-identification placeholders such as `[**Known lastname**]` are removed, as they carry no signal.
- Text is lowercased.
- Alphabetic word tokens are kept; pure numbers and punctuation are dropped, which keeps the vocabulary small and
  readable.

There is a genuine trade-off here. Numbers such as a creatinine value carry clinical meaning, and a production
system might keep and bucket them. For a teaching baseline, a rule you can see through is preferred. The important
discipline is that the exact same cleaning feeds both models, so any difference in results comes from the model,
not from different text.

## How success is measured

Plain accuracy is misleading for this problem. Most of the 50 codes are absent from any given note, so a model
that predicts "no" for everything scores high per label while being useless. Three metrics are used instead.

- **Micro-F1** pools every label decision into one precision and recall. It is dominated by the frequent codes and
  reports how many code assignments were correct overall.
- **Macro-F1** computes F1 separately for each of the 50 codes and averages them with equal weight, so a rare code
  counts as much as a common one. It shows whether the model only learned the easy codes.
- **Precision@k** asks, of the k codes the model is most confident about, how many are actually present. This
  matches how the tool would be used: a coder is handed a short ranked list to confirm.

The decision threshold is chosen on the validation set, and results are reported once on a test set that is never
touched during development.

## The baseline model

Each note is represented by **TF-IDF**: term frequency weighted down by how common a word is across all notes, so
distinctive terms like *dialysis* outweigh *the* or *patient*. A separate **logistic regression is trained for each
code** in a one-vs-rest scheme, giving 50 independent yes/no classifiers. The regularization strength is selected
on the validation set.

This is intentionally old-fashioned. It is fast and transparent, and on frequent codes it is a strong opponent,
which is exactly why it is built first: it is the bar the neural network has to clear.

## The deep model

The deep model is a convolutional network with per-label attention, the **CAML** architecture (Mullenbach et al.,
2018). The data flows through four stages:

```
tokens  ->  word embeddings  ->  1-D convolution  ->  per-label attention  ->  50 code probabilities
```

- **Embeddings** map each word to a learned vector, so the model can discover that *renal* and *kidney* are
  related rather than treating them as unrelated symbols.
- **A 1-D convolution** slides a window of 10 words across the note. It acts as a learned phrase detector, firing
  on local patterns such as *acute renal failure* wherever they occur.
- **Per-label attention** is the central idea. Instead of compressing the note into a single summary, the model
  keeps a separate attention distribution for each of the 50 codes, so each code decides for itself which parts of
  the note to focus on. The heart-failure code attends to different words than the diabetes code.
- Each code's focused summary passes through a **sigmoid** to produce its probability.

Configuration used here: embedding dimension 100, 128 convolutional filters of width 10, a vocabulary of about
29,000 words built from the training split only, a maximum input length of 1,500 words, batch size 32, and the
Adam optimizer at learning rate 1e-3, trained for 15 epochs. It is a genuine convolutional network, yet it trains
from scratch on a CPU in about 16 minutes.

## Results

Held-out test set (patient-disjoint), 50 most frequent ICD-9 codes:

| Metric | TF-IDF + Logistic Regression | CNN + attention |
|---|---|---|
| Micro-F1 | 0.602 | **0.614** |
| Macro-F1 | **0.533** | 0.515 |
| Precision@5 | 0.520 | **0.529** |
| Precision@8 | 0.408 | 0.408 |

![Baseline vs CNN on the test set](results/fig_compare.png)

Read honestly, the two models are close on accuracy. The CNN edges the baseline on micro-F1 and precision@5, the
metrics that track the frequent codes and the quality of the top few suggestions. The tuned linear baseline keeps
a slight edge on macro-F1, which weights every code equally, a fair result given that the CNN reads only the first
1,500 words of each note on a benchmark-sized sample. The point is not that one model wins the scoreboard. It is
that a well-tuned bag-of-words model is a strong opponent, and the CNN adds something the baseline cannot: for each
predicted code, it highlights the phrase in the note it attended to.

## Reading the model's evidence

Per-label attention is what makes the CNN's predictions inspectable. For any note, each predicted code exposes the
window of words its attention concentrated on. In practice these windows line up with clinical intuition: a
hypothyroidism prediction tends to attend to a mention of levothyroxine, its standard replacement therapy; a
chronic-kidney-disease prediction attends to phrases naming renal injury; a hyperlipidemia prediction attends to
the term itself in the past-medical-history section.

This matters for two reasons. First, a coder can confirm or reject a suggestion in seconds by reading the phrase
behind it. Second, when the model is wrong, the highlighted phrase usually shows why. In one case the model
predicted severe sepsis from language about fluid resuscitation and empiric treatment even though sepsis was not a
coded diagnosis for that admission, and the attended phrase makes that error immediately understandable. A model
whose evidence can be read is more useful in a clinical setting than a black box with the same score.

To respect the data use agreement, this repository does not reproduce real note excerpts. Run the notebook with
your own credentialed copy of MIMIC-III to see the attended phrases on real discharge summaries.

## Repository layout

```
ICD_Coding_Tutorial.ipynb   the tutorial: builds the cohort, trains both models, evaluates, and shows attention
ICD_Coding_Tutorial.pptx    slide presentation with detailed speaker notes
README.md                   this document
requirements.txt            Python dependencies
results/fig_compare.png     the baseline-vs-CNN comparison figure
data/                       not included; you place the three MIMIC-III CSVs here (git-ignored)
```

The notebook is stored without cell outputs that contain note text, in line with the data use agreement.

## Running it yourself

You need credentialed access to MIMIC-III on PhysioNet. Place the three CSV files in a `data/` folder next to the
notebook:

```
data/
  NOTEEVENTS.csv
  DIAGNOSES_ICD.csv
  D_ICD_DIAGNOSES.csv
```

Then install the dependencies and run the notebook top to bottom:

```bash
pip install -r requirements.txt
```

Open `ICD_Coding_Tutorial.ipynb` in Jupyter and run all cells. It builds the cohort, trains the baseline and the
CNN, evaluates both, and produces every figure.

## Reproducibility

The whole pipeline runs on an ordinary CPU in about 20 minutes: the cohort build scans the notes file once, the
baseline trains in roughly 5 minutes, and the CNN trains in about 16 minutes. Random seeds are fixed for NumPy,
Python, and PyTorch, and the data is split deterministically by patient, so repeated runs reproduce the same
cohort and the same reported numbers.

## Limitations

- **Scope.** The 50 most frequent codes on a benchmark-sized sample. Real coding spans thousands of codes,
  including rare ones that carry the most weight, and those are much harder.
- **Truncation.** The CNN reads the first 1,500 words of each note. Discharge summaries often place the discharge
  diagnosis list near the end, so some of the most predictive text can be discarded. Reading the full note is a
  clear next step.
- **Label quality.** ICD codes are billing labels entered by humans who disagree with one another. They are not
  perfect ground truth, which caps how well any model can score against them.
- **Intended use.** This is a first-pass suggestion tool for a human coder to confirm. It is not an autonomous
  biller and not a clinical device.
- **Generalization.** Everything here is MIMIC-III intensive-care notes from a single hospital. Another hospital
  or ward would require revalidation.

## Data use and privacy

MIMIC-III is governed by the PhysioNet Credentialed Health Data Use Agreement. This repository contains **no data
and no note text**: the `data/` folder is git-ignored, the notebook is stored without note-bearing outputs, and no
raw excerpts appear in the README or figures. To reproduce the results, obtain your own credentialed copy of
MIMIC-III from PhysioNet.

## Presentation

A slide presentation with detailed speaker notes accompanies this project: `ICD_Coding_Tutorial.pptx`. It walks
through the same material as this README, one step per slide. The worked attention example in the slides uses
illustrative phrases rather than real note text, in keeping with the data use agreement.

## Reference

James Mullenbach, Sarah Wiegreffe, Jon Duke, Jimeng Sun, and Jacob Eisenstein. *Explainable Prediction of Medical
Codes from Clinical Text.* NAACL 2018. (The CAML architecture used for the deep model.)
