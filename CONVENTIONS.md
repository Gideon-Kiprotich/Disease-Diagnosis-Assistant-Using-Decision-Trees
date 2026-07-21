# CONVENTIONS.md — Cancer Symptom Screener Project

Read this before making any changes. This file should be loaded into every
agent session (see "How to load this" at the bottom for your specific tool).

## What this project is
A Decision Tree classifier predicting one of 4 classes (Breast Cancer, Lung
Cancer, Cervical Cancer, No Red Flags) from patient symptoms + age + sex.
Course: APT3010A Intro to AI. Framed as a symptom-AWARENESS/triage tool,
NOT a diagnostic tool — keep that disclaimer in any user-facing text.

## Fixed files — do not regenerate or silently change
- `cancer_symptom_dataset.csv` — the dataset. Already built and validated.
- `generate_cancer_dataset.py` — the generator. If asked to change symptom
  probabilities or class list, confirm with me first — these were
  deliberately sourced from medical references and documented in my report.

## Coding standards
- Python, scikit-learn, pandas, matplotlib/seaborn. No new frameworks
  without asking first.
- Prioritize readable code over compressed/clever code — I need to be able
  to explain every line in a presentation.
- Comment the *why* behind a choice (e.g., why this `max_depth`), not just
  restating what the line does.
- After producing any result (accuracy, confusion matrix, feature
  importances), add a one-paragraph plain-language explanation of what it
  means. I will be presenting this, not just submitting code.

## Working style
- Do not silently make a design decision on my behalf (choice of
  hyperparameter range, evaluation metric, etc.) — flag it explicitly so I
  can decide whether to own it or change it.
- Small, reviewable steps over one large change. I want to be able to
  follow what happened.
- Use a separate git commit per meaningful change, with a clear message
  (see PROGRESS_TRACKING.md for the exact convention).

## Project phases (current status: update as you go)
- [ ] Phase 1: Load data, EDA, encode features, train/test split, baseline
      DecisionTreeClassifier
- [ ] Phase 2: GridSearchCV tuning, per-class precision/recall/F1,
      feature importance, plot_tree visualization, patient walkthroughs
- [ ] Phase 3: Report writing, presentation slides, optional Streamlit demo
- [ ] Phase 3 (stretch, optional): WBCD side comparison — see below

## Stretch addition: WBCD side comparison (decided — keep in scope)
A SEPARATE, independent binary decision tree trained on the real UCI
"Breast Cancer Wisconsin (Diagnostic)" dataset (malignant/benign from cell
nucleus measurements), scoped only to breast cancer. This is NOT merged
into the main 4-class symptom model — different feature schema entirely
(real clinical measurements vs. patient-reported symptoms).
- Load via `pip install ucimlrepo` -> `fetch_ucirepo(id=17)` — no missing
  values, minimal cleaning needed.
- Purpose: contrast a symptom-based screening tree (can only flag "go get
  checked") against a measurement-based diagnostic tree (can approach true
  diagnostic accuracy) — this contrast is the intended insight, not a
  head-to-head "which model wins."
- Scope: one binary tree, one comparison chart, ~half a page of report
  discussion. Do not let this expand into a second full pipeline.

## Environment: hybrid local dev + Colab execution
Coding happens locally (VS Code + Continue/Cline/Aider + Ollama). Colab is
used only to run and submit the final notebook. Practical implications:
- Prefer building the analysis as notebook cells (or `.py` logic split
  into cells at the end) rather than a script with local file paths that
  won't exist in a fresh Colab runtime.
- Cell 1 of the final notebook should regenerate the dataset from
  `generate_cancer_dataset.py`'s logic (it's seeded, so output is
  identical every run) rather than assuming an uploaded CSV — this keeps
  the notebook fully self-contained and reproducible for submission.
- Before submission: always test with `Runtime > Run all` on a fresh
  Colab runtime, not just re-running cells in an already-populated session.

## How to load this file into your agent
- **Aider**: `aider --read CONVENTIONS.md` or add `read: CONVENTIONS.md` to
  `.aider.conf.yml`
- **Cline**: copy relevant sections into `.clinerules/general.md`
- **Continue.dev**: copy relevant sections into `.continue/rules/general.md`
