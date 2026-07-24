# Repository Guidelines

## Project Structure & Module Organization

- `phase_1_baseline.py` contains exploratory data analysis, preprocessing, the
  stratified train/test split, and the baseline decision-tree model.
- `phase_2_tuning.py` performs GridSearchCV tuning, evaluation, error analysis,
  decision-path walkthroughs, and visualization generation.
- `cancer_symptom_dataset.csv` is the checked-in input dataset. Treat it as a
  validated project input; do not casually regenerate or alter it.
- `tuned_tree.png` and `feature_importance.png` are generated result artifacts.
- `CONVENTIONS.md` contains project context, modelling constraints, and working
  preferences. Read it before changing analysis code. There is currently no
  dedicated `tests/` directory.

## Build, Test, and Development Commands

This project has no build system or dependency file. With Python and the
project dependencies installed, run the phase scripts from the repository root:

```bash
python phase_1_baseline.py   # Run EDA and the baseline tree
python phase_2_tuning.py     # Tune, evaluate, and generate PNG artifacts
```

The scripts use pandas, NumPy, scikit-learn, Matplotlib, and seaborn. Run
`python --version` and install missing packages in an isolated environment.
There is no automated test command; validate changes by running the affected
phase and checking its metrics, printed interpretations, and output images.

## Coding Style & Naming Conventions

Use readable Python with four-space indentation, descriptive `snake_case`
variables and functions, and `PascalCase` only for classes. Keep the existing
cell markers (`# %%`) and explanatory comments. Explain why modelling choices
are made, not merely what each line does. Preserve reproducibility through the
existing `random_state=42` and compatible data-processing steps.

## Testing Guidelines

No test framework or coverage threshold is configured. For analysis changes,
run the relevant script end-to-end, confirm it exits successfully, and compare
train/test metrics and classification reports with the intended behaviour.
For changes affecting evaluation, inspect the confusion matrix and generated
visualizations as well.

## Commit & Pull Request Guidelines

Use short, specific commit subjects describing one meaningful change, such as
`feat: add baseline evaluation` or `Fix phase 2 tuning`. Keep commits focused.
Pull requests should explain the analytical change, dataset or metric impact,
and validation performed; include screenshots when visualization output changes
and call out any assumptions or model-selection decisions. Preserve the
project’s symptom-awareness disclaimer: this is not a diagnostic tool.

## Security & Configuration Tips

Do not commit credentials, private patient data, or unreviewed dataset changes.
Keep generated outputs reproducible and avoid changing the dataset schema,
target classes, or medical assumptions without documenting and reviewing them.
