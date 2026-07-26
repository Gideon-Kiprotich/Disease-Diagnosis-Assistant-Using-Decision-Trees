# %% [markdown]
# # Final Consolidated Colab Notebook
#
# This notebook combines the reproducible dataset generation, Phase 1 baseline,
# Phase 2 tuning and interface, v2 ablation generation, and subgroup analysis
# in the requested top-to-bottom order.

# %%
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# %% [markdown]
# ## 1. Original Dataset Generation
#
# This section recreates the original synthetic dataset in-session before the
# modelling phases, so the notebook does not depend on a user-uploaded CSV.

# %%
"""
Synthetic Breast / Lung / Cervical Cancer Warning-Sign Screener
Dataset Generator
------------------------------------------------------------------
Builds a synthetic (but medically-grounded) dataset mapping patient
symptoms to a "likely area of concern" label, for use with a
Decision Tree classifier.

IMPORTANT MEDICAL/ETHICAL FRAMING (keep this in your report):
This models a SYMPTOM-AWARENESS / "when to see a doctor" triage tool,
NOT a diagnostic tool. Real cancer diagnosis requires imaging, biopsy,
or lab work -- no symptom checklist can confirm cancer. Many cancers
are also asymptomatic in early stages; this dataset only models
patients who are already symptomatic. A "No Red Flags" class is
included specifically so the model isn't forced to diagnose every
symptom combination as cancer.

Symptom probabilities per condition were derived from published
symptom descriptions (Mayo Clinic, American Cancer Society, Cleveland
Clinic). These are reasoned estimates translated from qualitative
descriptions ("hallmark", "advanced-stage only", "sometimes") into
probabilities -- NOT clinical measurements. Document this translation
and your reasoning in your report.

Usage:
    python generate_cancer_dataset.py

Produces:
    cancer_symptom_dataset.csv
"""

# ---------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------
# 2. Shared symptom vocabulary, grouped by body system for clarity
# ---------------------------------------------------------------
GENERAL = ["unexplained_weight_loss", "fatigue", "night_sweats", "persistent_pain"]
BREAST = ["breast_lump", "breast_skin_changes", "nipple_discharge",
          "nipple_inversion", "breast_swelling"]
LUNG = ["persistent_cough", "coughing_blood", "chest_pain",
        "shortness_of_breath", "hoarseness"]
CERVICAL = ["abnormal_vaginal_bleeding", "unusual_vaginal_discharge", "pelvic_pain"]

SYMPTOMS = GENERAL + BREAST + LUNG + CERVICAL   # 17 symptoms total

# Biological constraint -- forced to 0 for male patients regardless of
# sampled probability or noise (cervical cancer requires a cervix).
FEMALE_ONLY = ["abnormal_vaginal_bleeding", "unusual_vaginal_discharge"]


def make_profile(overrides: dict, default: float = 0.02) -> dict:
    """Start every symptom at a low background rate, then override
    with the disease's actual hallmark/associated symptoms."""
    profile = {s: default for s in SYMPTOMS}
    profile.update(overrides)
    return profile


# ---------------------------------------------------------------
# 3. Condition profiles: symptom probabilities + realistic sex mix
# ---------------------------------------------------------------
DISEASE_PROFILES = {
    "Breast Cancer": {
        "sex_distribution": {"F": 0.99, "M": 0.01},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.15, "fatigue": 0.25, "night_sweats": 0.10,
            "persistent_pain": 0.20, "breast_lump": 0.85, "breast_skin_changes": 0.35,
            "nipple_discharge": 0.20, "nipple_inversion": 0.15, "breast_swelling": 0.30,
        }),
    },
    "Lung Cancer": {
        "sex_distribution": {"F": 0.45, "M": 0.55},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.40, "fatigue": 0.35, "night_sweats": 0.15,
            "persistent_pain": 0.20, "persistent_cough": 0.75, "coughing_blood": 0.35,
            "chest_pain": 0.40, "shortness_of_breath": 0.45, "hoarseness": 0.25,
        }),
    },
    "Cervical Cancer": {
        "sex_distribution": {"F": 1.0},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.15, "fatigue": 0.25, "night_sweats": 0.08,
            "persistent_pain": 0.15, "abnormal_vaginal_bleeding": 0.75,
            "unusual_vaginal_discharge": 0.55, "pelvic_pain": 0.50,
        }),
    },
    "No Red Flags (Routine/Benign)": {
        "sex_distribution": {"F": 0.50, "M": 0.50},
        "symptoms": make_profile({
            "fatigue": 0.30, "persistent_pain": 0.10,
            "unexplained_weight_loss": 0.03, "night_sweats": 0.05,
        }, default=0.015),
    },
}

# ---------------------------------------------------------------
# 4. Generation settings
# ---------------------------------------------------------------
SAMPLES_PER_CLASS = 300   # -> 1,200 total rows (balanced classes)
NOISE_PROB = 0.03         # small chance any given symptom randomly flips,
                           # modeling atypical presentation / comorbidities


def generate_patient(disease: str, profile: dict, patient_id: int) -> dict:
    sexes = list(profile["sex_distribution"].keys())
    probs = list(profile["sex_distribution"].values())
    sex = rng.choice(sexes, p=probs)
    age = int(rng.integers(20, 90))

    row = {"patient_id": patient_id, "age": age, "sex": sex}
    for symptom in SYMPTOMS:
        p = profile["symptoms"][symptom]
        present = rng.random() < p
        if rng.random() < NOISE_PROB:
            present = not present
        if symptom in FEMALE_ONLY and sex != "F":   # enforce biological plausibility
            present = False
        row[symptom] = int(present)
    row["diagnosis"] = disease
    return row


def main():
    rows = []
    patient_id = 1
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(SAMPLES_PER_CLASS):
            rows.append(generate_patient(disease, profile, patient_id))
            patient_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    out_path = "cancer_symptom_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass balance:")
    print(df["diagnosis"].value_counts())
    print("\nSex distribution by diagnosis (sanity check for biological plausibility):")
    print(pd.crosstab(df["diagnosis"], df["sex"]))
    print("\nHallmark symptom prevalence by diagnosis (sanity check):")
    hallmark_cols = ["breast_lump", "persistent_cough", "abnormal_vaginal_bleeding"]
    print(df.groupby("diagnosis")[hallmark_cols].mean().round(2))


if __name__ == "__main__":
    main()
# %% [markdown]
# ## 2. Phase 1 Baseline Pipeline
#
# This section performs the original exploratory analysis, preprocessing,
# stratified split, baseline Decision Tree training, and evaluation.

# %% [markdown]
# ### Integration flag
#
# The consolidated notebook now writes `cancer_symptom_dataset.csv`, matching
# the filename loaded by Phase 1 and Phase 2. The earlier filename mismatch
# has been resolved explicitly here.

# %% [markdown]
# # Phase 1: EDA, Preprocessing, and Baseline Decision Tree Classifier
# **Course:** APT3010A Intro to AI - Cancer Symptom Screener Project
# 
# This file serves as the pipeline for Phase 1. It is formatted with `# %%` cell delimiters 
# so it can be run either as a standard Python script or as interactive notebook cells in VS Code/Google Colab.

# %%
# Configure pandas display settings for complete console printouts
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# %% [markdown]
# ## Step 1: Load Data and Drop Non-Predictive Features

# %%
print("=== Step 1: Loading Dataset ===")
# We load the verified cancer symptom dataset.
# The dataset contains patient-reported symptoms, demographic details (age, sex), and target diagnosis.
data_path = "cancer_symptom_dataset.csv"
df = pd.read_csv(data_path)
print(f"Original dataset shape: {df.shape}")

# RATIONALE & JUDGMENT CALL:
# We drop 'patient_id' because it is a randomly generated unique identifier. It does not carry any
# predictive signal or clinical significance. If we kept it, a Decision Tree classifier might split on it,
# leading to severe overfitting (learning specific patient IDs instead of generalizable symptom patterns).
df_clean = df.drop(columns=["patient_id"])
print("Dropped 'patient_id'. Cleaned dataset shape:", df_clean.shape)
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 2: Exploratory Data Analysis (EDA)

# %%
print("=== Step 2: Exploratory Data Analysis ===")

# A. Confirm Class Balance for Target ('diagnosis')
# RATIONALE:
# Checking class balance helps determine if there is a majority class that could bias the classifier.
# If one class is dominant, the model could achieve high accuracy by simply predicting the majority class.
print("--- Class Balance for Diagnosis ---")
class_counts = df_clean["diagnosis"].value_counts()
class_percentages = df_clean["diagnosis"].value_counts(normalize=True) * 100
class_balance_df = pd.DataFrame({
    "Count": class_counts,
    "Percentage (%)": class_percentages
})
print(class_balance_df)
class_balance_summary = ", ".join(
    f"{diagnosis}: {percentage:.1f}%"
    for diagnosis, percentage in class_percentages.items()
)
print(
    f"\n[EDA Interpretation - Class Balance]: The dataset contains "
    f"{class_percentages.size} diagnosis categories with class shares of "
    f"{class_balance_summary}. This means no category is represented by a "
    f"dominant majority of the data."
)
print()

# B. Symptom Prevalence Grouped by Diagnosis
# RATIONALE:
# We calculate the mean of each binary symptom column (0 = absent, 1 = present) grouped by diagnosis.
# This represents the proportion of patients in each diagnosis group who report that symptom.
# It allows us to verify if the symptoms align with clinical expectations (e.g. breast lumps corresponding to breast cancer).
print("--- Symptom Prevalence by Diagnosis (Means) ---")
# Select all symptom columns (exclude age, sex, and diagnosis)
symptom_cols = [col for col in df_clean.columns if col not in ["age", "sex", "diagnosis"]]
prevalence_by_diag = df_clean.groupby("diagnosis")[symptom_cols].mean().T
print(prevalence_by_diag)
top_symptoms_by_diagnosis = {
    diagnosis: (
        prevalence_by_diag[diagnosis].idxmax(),
        prevalence_by_diag[diagnosis].max() * 100,
    )
    for diagnosis in prevalence_by_diag.columns
}
breast_symptom, breast_prevalence = top_symptoms_by_diagnosis["Breast Cancer"]
cervical_symptom, cervical_prevalence = top_symptoms_by_diagnosis["Cervical Cancer"]
lung_symptom, lung_prevalence = top_symptoms_by_diagnosis["Lung Cancer"]
print(
    f"\n[EDA Interpretation - Symptom Prevalence]: The most prevalent symptom "
    f"for Breast Cancer is '{breast_symptom}' ({breast_prevalence:.1f}%), "
    f"for Cervical Cancer is '{cervical_symptom}' ({cervical_prevalence:.1f}%), "
    f"and for Lung Cancer is '{lung_symptom}' ({lung_prevalence:.1f}%). "
    f"These class-specific patterns can help distinguish the categories."
)
print()

# C. Crosstab of Diagnosis by Sex
# RATIONALE:
# We cross-tabulate diagnosis against sex. Medically, certain cancers have strict biological links:
# for instance, cervical cancer should only occur in biological females (F) in the dataset.
# Verifying this distribution ensures there are no demographic anomalies in our data generator.
print("--- Crosstab of Diagnosis by Sex ---")
crosstab_sex = pd.crosstab(df_clean["diagnosis"], df_clean["sex"])
print(crosstab_sex)
cervical_female_percentage = (
    crosstab_sex.loc["Cervical Cancer", "F"]
    / crosstab_sex.loc["Cervical Cancer"].sum()
    * 100
)
breast_female_percentage = (
    crosstab_sex.loc["Breast Cancer", "F"]
    / crosstab_sex.loc["Breast Cancer"].sum()
    * 100
)
print(
    f"\n[EDA Interpretation - Demographic Distribution]: The crosstab shows "
    f"that {cervical_female_percentage:.1f}% of Cervical Cancer records are "
    f"female, while {breast_female_percentage:.1f}% of Breast Cancer records "
    f"are female. These distributions preserve the expected demographic "
    f"relationships in the dataset."
)
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 3: Encode Categorical Features and Define X and y

# %%
print("=== Step 3: Encoding and Feature Definition ===")
# RATIONALE & JUDGMENT CALL:
# scikit-learn's DecisionTreeClassifier implementation requires all input features to be numerical.
# We encode the 'sex' column as a binary feature: Male (M) -> 0, Female (F) -> 1.
# This is a simple binary encoding that maintains the single-column representation without adding dimensionality.
df_clean["sex_encoded"] = df_clean["sex"].map({"M": 0, "F": 1})

# Guardrail: Check if there are any unmapped/NaN values after encoding
if df_clean["sex_encoded"].isnull().any():
    raise ValueError("Error: Found unmapped or NaN values in the 'sex' column after mapping!")

# Define X (features) and y (target)
# Features include the age, the binary encoded sex, and all binary symptom columns.
feature_cols = symptom_cols + ["age", "sex_encoded"]
X = df_clean[feature_cols]
y = df_clean["diagnosis"]

print(f"Features (X) shape: {X.shape}")
print(f"Features list: {list(X.columns)}")
print(f"Target (y) shape: {y.shape}")
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 4: Stratified Train/Test Split

# %%
print("=== Step 4: Stratified Train/Test Split ===")
# RATIONALE & JUDGMENT CALL:
# We perform an 80/20 train/test split.
# Crucially, we set `stratify=y` to ensure that the class distribution of the target ('diagnosis')
# is identical in both the training set and the test set.
#
# WHY STRATIFIED IS CRITICAL HERE:
# 1. Class Representative Sample: With four perfectly balanced classes (25% each), a stratified split
#    guarantees that exactly 25% of both training (768 samples) and test (240 samples) data belong to each class.
#    A standard random split might lead to a class being underrepresented in the training set or overrepresented in the test set.
# 2. Gender-Locked Conditions: Since cervical cancer is biologically locked to females (100% Female) and breast cancer
#    is heavily female-skewed (98.7% Female in this dataset), non-stratified splitting risks creating sub-splits 
#    lacking sufficient representative patterns for these specific subgroups, hurting the tree's split rules.
#
# We set `random_state=42` to ensure that the partition is completely reproducible across runs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\n--- Training Set Class Distribution (%) ---")
print(y_train.value_counts(normalize=True) * 100)
print("\n--- Test Set Class Distribution (%) ---")
print(y_test.value_counts(normalize=True) * 100)
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 5: Train and Evaluate Baseline Decision Tree Classifier

# %%
print("=== Step 5: Baseline Decision Tree Training & Evaluation ===")
# RATIONALE & JUDGMENT CALL:
# We train a default DecisionTreeClassifier. By default, scikit-learn does not limit the depth (max_depth=None),
# meaning the tree will grow until all leaves are pure or contain fewer than min_samples_split (default 2) samples.
# We set `random_state=42` to ensure that the choice of feature splits when multiple features offer the same
# information gain (entropy decrease) is deterministic and reproducible.
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
gap = (train_accuracy - accuracy) * 100

print(f"Baseline Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Baseline Test Accuracy: {accuracy * 100:.2f}%")
print(f"Train/Test Gap: {gap:.2f} percentage points")

# Generate and print the confusion matrix
classes = clf.classes_
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

print("\n--- Confusion Matrix ---")
print(cm_df)

# Print a detailed classification report containing precision, recall, and F1-score per class
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n--- Plain-Language Baseline Results Interpretation ---")
print(
    f"Train vs. Test Gap (Overfitting): The unconstrained baseline decision tree "
    f"achieves training accuracy of {train_accuracy * 100:.2f}% but drops to "
    f"{accuracy * 100:.2f}% on the unseen test set. This {gap:.2f} percentage "
    f"point gap suggests that an unpruned tree memorizes training noise and "
    f"patient-specific nuances, demonstrating the need for hyperparameter "
    f"tuning to enforce tree regularization.\n"
    f"Confusion Matrix Insights: The baseline model performs well on distinct "
    f"anchor symptoms, but exhibits notable misclassifications when "
    f"non-specific symptoms overlap with Routine/Benign cases."
)

# %% [markdown]
# ## 3. Phase 2 Tuning, Phase 2b, Walkthroughs, and Interface
#
# This section preserves the complete Phase 2 sequence: accuracy-based tuning,
# evaluation, the shared decision-path helper and walkthroughs, cost-sensitive
# Phase 2b model comparison, and the interactive screening interface.

# %% [markdown]
# # Phase 2: Hyperparameter Tuning, Evaluation, and Patient Walkthroughs
# **Course:** APT3010A Intro to AI - Cancer Symptom Screener Project
# 
# This file completes Phase 2 of the pipeline. It is formatted with `# %%` cell delimiters 
# so it can be run either as a standard Python script or as interactive notebook cells in VS Code/Google Colab.

# %%
# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Setup: Load Dataset and Reproduce Phase 1 Train/Test Split
# RATIONALE: We use the exact same data cleaning, encoding (`sex_encoded`), and stratified 80/20 train/test split (`random_state=42`) from Phase 1 to ensure direct comparability across model versions.

# %%
print("=== Phase 2 Setup: Reusing Phase 1 Dataset & Train/Test Split ===")
# Reuse the dataframe loaded by Phase 1; the standalone Phase 2 file reloaded this same CSV.
df_clean = df.drop(columns=["patient_id"])
df_clean["sex_encoded"] = df_clean["sex"].map({"M": 0, "F": 1})

symptom_cols = [col for col in df_clean.columns if col not in ["age", "sex", "diagnosis", "sex_encoded"]]
feature_cols = symptom_cols + ["age", "sex_encoded"]
X = df_clean[feature_cols]
y = df_clean["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 1: GridSearchCV Hyperparameter Tuning
# 
# ### JUDGMENT CALL & RATIONALE FOR PARAMETER RANGES:
# 1. `max_depth`: Range `[3, 4, 5, 6, 7, 8, 10, None]`
#    - **Why**: The unconstrained Phase 1 baseline (`max_depth=None`) grew deep trees that memorized training noise (99.90% train accuracy vs 77.92% test accuracy). Searching depths from 3 to 10 explores the spectrum between simple interpretable rules and complex pattern matching, while `None` acts as an anchor control.
# 2. `min_samples_leaf`: Range `[1, 2, 5, 10, 15, 20]`
#    - **Why**: Requiring a minimum number of samples in leaf nodes prevents the tree from creating isolated split rules for 1 or 2 outlier patients, forcing leaves to reflect statistically meaningful symptom groups.
# 3. `criterion`: `['gini', 'entropy']`
#    - **Why**: Gini impurity measures variance reduction while Entropy measures information gain (logarithmic). Comparing both tests whether information entropy handles high-dimensional binary symptom presence splits better than Gini.
# 4. `cv=5`: 5-Fold Stratified Cross-Validation
#    - **Why**: Evaluates each hyperparameter combination across 5 distinct validation folds to avoid overfitting to a single train/validation partition.

# %%
print("=== Step 1: Running GridSearchCV ===")

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, None],
    'min_samples_leaf': [1, 2, 5, 10, 15, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n--- GridSearchCV Results ---")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best 5-Fold CV Mean Accuracy: {grid_search.best_score_ * 100:.2f}%")
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 2: Retrain Tuned Tree & Overfitting Gap Comparison

# %%
print("=== Step 2: Retrain Best Estimator & Compare Overfitting Gap ===")

# Retrain unconstrained baseline tree for fresh, dynamic comparison
clf_baseline = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf_baseline.fit(X_train, y_train)

train_acc_base = accuracy_score(y_train, clf_baseline.predict(X_train))
test_acc_base = accuracy_score(y_test, clf_baseline.predict(X_test))
gap_base = (train_acc_base - test_acc_base) * 100

# Retrieve best model
clf_best = grid_search.best_estimator_

# Predict on train and test sets
y_train_pred_tuned = clf_best.predict(X_train)
y_test_pred_tuned = clf_best.predict(X_test)

# Calculate accuracies
train_acc_tuned = accuracy_score(y_train, y_train_pred_tuned)
test_acc_tuned = accuracy_score(y_test, y_test_pred_tuned)
gap_tuned = (train_acc_tuned - test_acc_tuned) * 100

print(f"Tuned Train Accuracy: {train_acc_tuned * 100:.2f}%  (Baseline: {train_acc_base * 100:.2f}%)")
print(f"Tuned Test Accuracy:  {test_acc_tuned * 100:.2f}%  (Baseline: {test_acc_base * 100:.2f}%)")
print(f"Tuned Train/Test Gap: {gap_tuned:.2f} percentage points  (Baseline Gap: {gap_base:.2f} percentage points)")

gap_reduction = gap_base - gap_tuned
print(f"\nResult: Hyperparameter tuning reduced the overfitting gap by {gap_reduction:.2f} percentage points!")
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 3: Per-Class Evaluation & Clinical False-Negative Analysis

# %%
print("=== Step 3: Classification Report & False-Negative Clinical Analysis ===")

classes = clf_best.classes_
cm_tuned = confusion_matrix(y_test, y_test_pred_tuned, labels=classes)
cm_tuned_df = pd.DataFrame(cm_tuned, index=classes, columns=classes)

print("--- Confusion Matrix (Tuned Model) ---")
print(cm_tuned_df)
print()

print("--- Classification Report (Tuned Model) ---")
report_text = classification_report(y_test, y_test_pred_tuned, target_names=classes)
print(report_text)

# Analyze False Negatives (actual Cancer patients misclassified as "No Red Flags (Routine/Benign)")
# vs False Positives (actual "No Red Flags" misclassified as a Cancer)
cancer_classes = [c for c in classes if c != "No Red Flags (Routine/Benign)"]

false_negatives_cancer = 0  # Cancer patient told "No Red Flags"
false_positives_benign = 0  # Benign patient told they have Cancer

for c in cancer_classes:
    false_negatives_cancer += cm_tuned_df.loc[c, "No Red Flags (Routine/Benign)"]

for c in cancer_classes:
    false_positives_benign += cm_tuned_df.loc["No Red Flags (Routine/Benign)", c]

print("--- Clinical Error Breakdown ---")
print(f"False Negatives (Cancer patients incorrectly classified as 'No Red Flags'): {false_negatives_cancer}")
print(f"False Positives (Routine/Benign patients incorrectly flagged for Cancer): {false_positives_benign}")
print("\n--- Clinical Asymmetry & Error Context ---")
print(
    "Clinical Explanation: In a cancer symptom screening tool, false negatives and false positives carry\n"
    "vastly different real-world risks. A False Negative (misclassifying a real cancer patient as 'No Red Flags')\n"
    "is catastrophic because it falsely reassures the patient, delaying crucial diagnostic workups and early intervention\n"
    "when disease treatability is highest. Conversely, a False Positive (flagging a benign case for cancer evaluation)\n"
    "merely triggers a secondary clinical check-up; while it may cause transient anxiety, it does not pose a\n"
    "life-threatening clinical risk. Therefore, minimizing false negatives is the primary clinical safety objective."
)
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Step 4: Visualizations (Decision Tree Structure & Feature Importances)

# %%
print("=== Step 4: Generating Visualizations ===")

# 4a: plot_tree Visualization
plt.figure(figsize=(20, 10), dpi=300)
# If the tuned tree is deep, we constrain plot_tree rendering depth to max_depth=3 for optimal readability.
render_depth = 3 if clf_best.get_depth() > 3 else clf_best.get_depth()
print(f"Rendering plot_tree up to depth={render_depth} (actual tree depth: {clf_best.get_depth()}) to maintain visual clarity.")

plot_tree(
    clf_best,
    max_depth=render_depth,
    feature_names=feature_cols,
    class_names=classes,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(f"Tuned Decision Tree Structure (Top {render_depth} Levels)", fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("tuned_tree.png", dpi=300)
plt.show()
plt.close()

# 4b: Feature Importance Bar Chart
plt.figure(figsize=(10, 6), dpi=300)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(
    data=importance_df,
    x='Importance',
    y='Feature',
    hue='Feature',
    legend=False,
    palette="viridis"
)
plt.title("Decision Tree Feature Importances (Gini Impurity / Entropy Decrease)", fontsize=14, fontweight='bold')
plt.xlabel("Relative Feature Importance Score", fontsize=12)
plt.ylabel("Symptom / Demographic Feature", fontsize=12)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
plt.close()

print("Saved visualization artifacts: 'tuned_tree.png' and 'feature_importance.png'.")
print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Shared Decision-Path Tracing Helper
#
# Keeping this traversal in one function ensures the walkthroughs and the
# interactive interface explain model decisions using the same tree logic.

# %%
def trace_decision_path(model, patient_row, feature_cols, classes):
    """Return decision steps and the final class predicted for one patient."""
    node_indicator = model.decision_path(patient_row)
    leaf_id = model.apply(patient_row)[0]
    node_indices = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    tree_ = model.tree_
    patient_data = patient_row.iloc[0]
    steps = []
    for node_id in node_indices:
        if node_id == leaf_id:
            continue

        feature_index = tree_.feature[node_id]
        feature_name = feature_cols[feature_index]
        patient_value = patient_data[feature_name]
        threshold = tree_.threshold[node_id]
        direction = "left" if patient_value <= threshold else "right"
        steps.append((feature_name, patient_value, threshold, direction))

    final_predicted_class = model.predict(patient_row)[0]
    return steps, final_predicted_class

# %% [markdown]
# ## Step 5: Patient Walkthroughs & Decision Path Traversal
# RATIONALE: To demonstrate clinical transparency and interpretability, we select 3 correctly classified patients from the test set (one each for Breast Cancer, Lung Cancer, and Cervical Cancer) and trace the step-by-step decision rules traversed by the tree.

# %%
print("=== Step 5: Patient Walkthroughs & Decision Paths ===")

# Create an evaluation dataframe for test set predictions
test_eval = X_test.copy()
test_eval["true_diagnosis"] = y_test
test_eval["predicted_diagnosis"] = y_test_pred_tuned

target_cancers = ["Breast Cancer", "Lung Cancer", "Cervical Cancer"]

for cancer in target_cancers:
    # Find matching correctly classified patients
    correct_matches = test_eval[
        (test_eval["true_diagnosis"] == cancer) & 
        (test_eval["predicted_diagnosis"] == cancer)
    ]
    
    if correct_matches.empty:
        print(f"No correctly classified patient found for {cancer}.")
        continue
        
    # Select the first representative patient
    patient_idx = correct_matches.index[0]
    patient_row = X_test.loc[[patient_idx]]
    patient_data = patient_row.iloc[0]
    
    print(f"\n" + "*"*60)
    print(f"PATIENT CASE STUDY: {cancer.upper()} (Test Sample Index: {patient_idx})")
    print("*"*60)
    print(f"True Diagnosis:      {cancer}")
    print(f"Predicted Diagnosis: {test_eval.loc[patient_idx, 'predicted_diagnosis']}")
    print("\nPatient Demographic & Symptom Profile:")
    present_symptoms = []
    for col in feature_cols:
        val = patient_data[col]
        if col == "sex_encoded":
            print(f"  - Sex: {'Female (1)' if val == 1 else 'Male (0)'}")
        elif col == "age":
            print(f"  - Age: {val} years")
        else:
            if val == 1:
                present_symptoms.append(col)
    print(f"  - Present Symptoms ({len(present_symptoms)}): {', '.join(present_symptoms) if present_symptoms else 'None'}")
    
    decision_steps, traced_prediction = trace_decision_path(
        clf_best,
        patient_row,
        feature_cols,
        classes,
    )
    print("\nDecision Tree Path Traversal:")
    for step_num, (feature_name, patient_value, threshold, direction) in enumerate(
        decision_steps,
        1,
    ):
        comparison = "<=" if direction == "left" else ">"
        print(
            f"  Step {step_num}: '{feature_name}' = {patient_value} "
            f"{comparison} {threshold:.2f}; go {direction}."
        )
    print(f"  Final Prediction -> '{traced_prediction}'")

    # Dynamic clinical rationale generated from the shared traced path.
    path_features = [step[0] for step in decision_steps]
    if path_features:
        if len(path_features) == 1:
            features_str = f"checking '{path_features[0]}'"
        elif len(path_features) == 2:
            features_str = f"first checking '{path_features[0]}', then '{path_features[1]}'"
        else:
            features_str = f"first checking '{path_features[0]}', then " + ", then ".join([f"'{f}'" for f in path_features[1:]])
        print(f"\nClinical Rationale: The tree reached this diagnosis by {features_str}.")

print("\n" + "="*50 + "\n")

# %% [markdown]
# ## Phase 2b: Cost-Sensitive Hyperparameter Tuning
#
# This section keeps the original tuned model intact and evaluates alternative
# models that assign greater cost to cancer cases predicted as No Red Flags.

# %%

CANCER_CLASSES = ["Breast Cancer", "Lung Cancer", "Cervical Cancer"]
BENIGN_CLASS = "No Red Flags (Routine/Benign)"

# This scorer counts a cancer-patient-classified-as-"No Red Flags" error as
# fn_penalty times worse than a normal error, instead of treating all mistakes
# equally like plain accuracy does.
def make_cost_sensitive_scorer(fn_penalty):
    def cost_sensitive_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct = (y_true == y_pred).sum()
        is_false_negative = np.isin(y_true, CANCER_CLASSES) & (y_pred == BENIGN_CLASS)
        penalty = is_false_negative.sum() * fn_penalty
        return (correct - penalty) / len(y_true)
    return make_scorer(cost_sensitive_score, greater_is_better=True)

# The classes are already balanced, so adding a 'balanced' class weight would
# duplicate the unweighted case. The explicit cancer weighting tests the
# intended cost-sensitive alternative.
cost_sensitive_param_grid = {
    **param_grid,
    "class_weight": [
        None,
        {
            "Breast Cancer": 2,
            "Lung Cancer": 2,
            "Cervical Cancer": 2,
            "No Red Flags (Routine/Benign)": 1,
        },
    ],
}

print("=== Phase 2b: Cost-Sensitive GridSearchCV ===")

# Preserve the original Phase 2 tuned model as the comparison point.
original_tuned_accuracy = test_acc_tuned
original_tuned_false_negatives = false_negatives_cancer

phase_2b_results = []
for fn_penalty in [1, 1.5, 2, 2.5, 3]:
    print(f"\n--- Running cost-sensitive search with fn_penalty={fn_penalty} ---")

    cost_sensitive_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=cost_sensitive_param_grid,
        cv=5,
        scoring=make_cost_sensitive_scorer(fn_penalty),
        n_jobs=-1,
        verbose=1,
    )
    cost_sensitive_search.fit(X_train, y_train)

    cost_sensitive_model = cost_sensitive_search.best_estimator_
    y_train_pred_cost_sensitive = cost_sensitive_model.predict(X_train)
    y_pred_cost_sensitive = cost_sensitive_model.predict(X_test)
    train_accuracy_cost_sensitive = accuracy_score(
        y_train,
        y_train_pred_cost_sensitive,
    )
    test_accuracy_cost_sensitive = accuracy_score(y_test, y_pred_cost_sensitive)
    train_test_gap_cost_sensitive = (
        train_accuracy_cost_sensitive - test_accuracy_cost_sensitive
    ) * 100

    cm_cost_sensitive = confusion_matrix(
        y_test,
        y_pred_cost_sensitive,
        labels=classes,
    )
    cm_cost_sensitive_df = pd.DataFrame(
        cm_cost_sensitive,
        index=classes,
        columns=classes,
    )

    # Reuse Phase 2 Step 3's cancer false-negative and benign false-positive
    # counting logic so the comparison uses the same error definitions.
    false_negatives_cost_sensitive = 0
    false_positives_cost_sensitive = 0

    for c in cancer_classes:
        false_negatives_cost_sensitive += cm_cost_sensitive_df.loc[
            c, BENIGN_CLASS
        ]

    for c in cancer_classes:
        false_positives_cost_sensitive += cm_cost_sensitive_df.loc[
            BENIGN_CLASS, c
        ]

    result = {
        "fn_penalty": fn_penalty,
        "best_params": cost_sensitive_search.best_params_,
        "model": cost_sensitive_model,
        "train_accuracy": train_accuracy_cost_sensitive,
        "test_accuracy": test_accuracy_cost_sensitive,
        "train_test_gap": train_test_gap_cost_sensitive,
        "false_negatives": false_negatives_cost_sensitive,
        "false_positives": false_positives_cost_sensitive,
    }
    phase_2b_results.append(result)

    print(f"Best params found: {result['best_params']}")
    print(f"Train accuracy: {result['train_accuracy'] * 100:.2f}%")
    print(f"Overall test accuracy: {result['test_accuracy'] * 100:.2f}%")
    print(f"Train/test gap: {result['train_test_gap']:.2f} percentage points")
    print(f"Cancer false negatives: {result['false_negatives']}")
    print(f"False positives: {result['false_positives']}")

fn_penalty_1_5_result = next(
    result for result in phase_2b_results if result["fn_penalty"] == 1.5
)
model = fn_penalty_1_5_result["model"]
print("\n--- Phase 2b Fitted Model Confirmation (fn_penalty=1.5) ---")
print(f"type(model): {type(model)}")
print(f"model.get_params(): {model.get_params()}")

# %% [markdown]
# ## Phase 2b Results: Accuracy and False-Negative Trade-off

# %%
penalty_values = [result["fn_penalty"] for result in phase_2b_results]
accuracy_values = [result["test_accuracy"] * 100 for result in phase_2b_results]
false_negative_values = [result["false_negatives"] for result in phase_2b_results]
original_tuned_accuracy_percentage = original_tuned_accuracy * 100
original_tuned_label = "Original tuned model (accuracy-only)"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    penalty_values,
    accuracy_values,
    marker="o",
    linewidth=2,
    label="Cost-sensitive model",
)
axes[0].axhline(
    y=original_tuned_accuracy_percentage,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=original_tuned_label,
)
axes[0].set_title("Overall Test Accuracy")
axes[0].set_xlabel("False-Negative Penalty")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_xticks(penalty_values)
axes[0].grid(True)
axes[0].legend()

axes[1].plot(
    penalty_values,
    false_negative_values,
    marker="o",
    linewidth=2,
    label="Cost-sensitive model",
)
axes[1].axhline(
    y=original_tuned_false_negatives,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=original_tuned_label,
)
axes[1].set_title("Cancer False-Negative Count")
axes[1].set_xlabel("False-Negative Penalty")
axes[1].set_ylabel("Count")
axes[1].set_xticks(penalty_values)
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Phase 2b Cost-Sensitive Tuning Trade-off")
fig.tight_layout()
fig.savefig("cost_sensitive_tradeoff.png", dpi=300)
plt.show()
plt.close(fig)

# Use the original tuned model as the baseline for a plain-language comparison.
first_false_negative_drop = next(
    (
        result
        for result in phase_2b_results
        if result["false_negatives"] < original_tuned_false_negatives
    ),
    None,
)
best_false_negative_result = min(
    phase_2b_results,
    key=lambda result: (result["false_negatives"], -result["test_accuracy"]),
)

print("\n--- Plain-Language Phase 2b Interpretation ---")
if first_false_negative_drop is None:
    print(
        f"Across the tested penalty values, the cancer false-negative count did "
        f"not drop below the original tuned model's count of "
        f"{original_tuned_false_negatives}."
    )
else:
    accuracy_cost = (
        original_tuned_accuracy - first_false_negative_drop["test_accuracy"]
    ) * 100
    print(
        f"The first observed drop in cancer false negatives occurs at a penalty "
        f"of {first_false_negative_drop['fn_penalty']}, decreasing the count from "
        f"{original_tuned_false_negatives} to "
        f"{first_false_negative_drop['false_negatives']}. The corresponding "
        f"accuracy changes from {original_tuned_accuracy * 100:.2f}% to "
        f"{first_false_negative_drop['test_accuracy'] * 100:.2f}%, a cost of "
        f"{accuracy_cost:.2f} percentage points."
    )

best_false_negative_count = best_false_negative_result["false_negatives"]
min_fn_penalty = best_false_negative_result["fn_penalty"]
plateau_results = [
    result
    for result in phase_2b_results
    if result["fn_penalty"] > min_fn_penalty
]
is_plateaued = len(plateau_results) > 0 and all(
    result["false_negatives"] == best_false_negative_count
    for result in plateau_results
)

if is_plateaued:
    accuracy_reference = (
        first_false_negative_drop["test_accuracy"]
        if first_false_negative_drop is not None
        else original_tuned_accuracy
    )
    accuracy_reference_label = (
        "the first reduction"
        if first_false_negative_drop is not None
        else "the original tuned model"
    )
    additional_accuracy_cost = (
        accuracy_reference - best_false_negative_result["test_accuracy"]
    ) * 100
    print(
        f"The lowest observed cancer false-negative count is "
        f"{best_false_negative_result['false_negatives']} at a penalty of "
        f"{best_false_negative_result['fn_penalty']}. Higher penalties do not "
        f"reduce that count further, indicating diminishing returns; they add "
        f"{additional_accuracy_cost:.2f} additional accuracy-loss percentage "
        f"points relative to {accuracy_reference_label}."
    )
else:
    print(
        f"The lowest observed cancer false-negative count is "
        f"{best_false_negative_result['false_negatives']} at a penalty of "
        f"{best_false_negative_result['fn_penalty']}. The tested range does not "
        f"yet show a clear point of diminishing returns."
    )

# %% [markdown]
# ## Interactive Symptom-Awareness Screening Interface
#
# This interface reuses the already-fitted Phase 2 models. It does not retrain
# either model, and the analysis-only decision-path explanation comes from the
# same `trace_decision_path` helper used by the patient walkthroughs.

# %%

SYMPTOM_GROUPS = {
    "General": [
        "unexplained_weight_loss",
        "fatigue",
        "night_sweats",
        "persistent_pain",
    ],
    "Breast": [
        "breast_lump",
        "breast_skin_changes",
        "nipple_discharge",
        "nipple_inversion",
        "breast_swelling",
    ],
    "Lung": [
        "persistent_cough",
        "coughing_blood",
        "chest_pain",
        "shortness_of_breath",
        "hoarseness",
    ],
    "Cervical": [
        "abnormal_vaginal_bleeding",
        "unusual_vaginal_discharge",
        "pelvic_pain",
    ],
}


def make_symptom_section(section_name, symptom_names):
    """Group related checkboxes so the form mirrors the symptom vocabulary."""
    section_checkboxes = []
    for symptom_name in symptom_names:
        checkbox = widgets.Checkbox(
            value=False,
            description=symptom_name.replace("_", " ").title(),
            indent=False,
            layout=widgets.Layout(width="320px"),
        )
        symptom_checkboxes[symptom_name] = checkbox
        section_checkboxes.append(checkbox)

    return widgets.VBox(
        [
            widgets.HTML(f"<b>{section_name} symptoms</b>"),
            *section_checkboxes,
        ],
        layout=widgets.Layout(
            border="1px solid #cccccc",
            padding="8px",
            margin="4px",
        ),
    )


# A single widget per symptom keeps the submitted patient profile explicit and
# prevents accidental feature values from being inferred from the UI layout.
symptom_checkboxes = {}
symptom_sections = [
    make_symptom_section(section_name, symptom_names)
    for section_name, symptom_names in SYMPTOM_GROUPS.items()
]

age_slider = widgets.IntSlider(
    value=40,
    min=20,
    max=89,
    step=1,
    description="Age:",
    continuous_update=False,
)
sex_dropdown = widgets.Dropdown(
    options=[("Female", "F"), ("Male", "M")],
    value="F",
    description="Sex:",
)
model_selector = widgets.ToggleButtons(
    options=["Standard Model", "High-Sensitivity Model"],
    value="Standard Model",
    description="Model:",
)
screening_button = widgets.Button(
    description="Get Screening Result",
    button_style="primary",
    icon="search",
)

# Retrieve the high-sensitivity model from the corrected Phase 2b results so
# the interface always uses the exact fitted model selected by that search.
high_sensitivity_result = next(
    result for result in phase_2b_results if result["fn_penalty"] == 1.5
)
high_sensitivity_model = high_sensitivity_result["model"]
standard_model = clf_best


def format_decision_path(decision_steps):
    """Turn shared tracer tuples into readable notebook Markdown."""
    if not decision_steps:
        return "No internal split was required for this patient profile."

    formatted_steps = []
    for step_number, (feature_name, patient_value, threshold, direction) in enumerate(
        decision_steps,
        1,
    ):
        comparison = "<=" if direction == "left" else ">"
        formatted_steps.append(
            f"{step_number}. `{feature_name}` = `{patient_value}` "
            f"{comparison} `{threshold:.2f}` → go **{direction}**"
        )
    return "\n".join(formatted_steps)


def build_screening_row():
    """Convert widget values into the exact feature schema used in training."""
    patient_values = {
        symptom_name: int(checkbox.value)
        for symptom_name, checkbox in symptom_checkboxes.items()
    }
    patient_values["age"] = age_slider.value
    patient_values["sex_encoded"] = 1 if sex_dropdown.value == "F" else 0
    return pd.DataFrame([patient_values], columns=feature_cols)


screening_output = widgets.Output()


def on_screening_button_clicked(_button):
    patient_row = build_screening_row()
    selected_model = (
        standard_model
        if model_selector.value == "Standard Model"
        else high_sensitivity_model
    )
    decision_steps, predicted_class = trace_decision_path(
        selected_model,
        patient_row,
        feature_cols,
        classes,
    )

    with screening_output:
        # Clear only the result area so the disclaimer and form remain visible.
        screening_output.clear_output(wait=True)
        display(Markdown(f"### Predicted class: **{predicted_class}**"))

        if model_selector.value == "High-Sensitivity Model":
            standard_prediction = standard_model.predict(patient_row)[0]
            display(
                Markdown(
                    "| Model | Prediction |\n"
                    "|---|---|\n"
                    f"| Standard Model | **{standard_prediction}** |\n"
                    f"| High-Sensitivity Model | **{predicted_class}** |"
                )
            )

        display(Markdown("#### Decision path\n" + format_decision_path(decision_steps)))


screening_button.on_click(on_screening_button_clicked)

screening_form = widgets.VBox(
    [
        widgets.HTML("<h3>Patient symptom profile</h3>"),
        *symptom_sections,
        widgets.HBox([age_slider, sex_dropdown]),
        model_selector,
        screening_button,
    ]
)
screening_disclaimer = widgets.HTML(
    "<b>This is a symptom-awareness screening aid, not a diagnosis. "
    "Consult a healthcare professional for any health concerns.</b>"
)

# The disclaimer is displayed outside the clearable Output widget so it is
# permanent and remains visible after every new screening result.
display(screening_form)
display(screening_disclaimer)
display(screening_output)

# %% [markdown]
# ## 4. v2 Controlled-Ablation Dataset Generation
#
# This section generates the benign-look-alike dataset and its analysis-only
# subpattern labels in-session without changing the original dataset.

# %%
"""
Synthetic Breast / Lung / Cervical Cancer Warning-Sign Screener
Dataset Generator
------------------------------------------------------------------
Builds a synthetic (but medically-grounded) dataset mapping patient
symptoms to a "likely area of concern" label, for use with a
Decision Tree classifier.

IMPORTANT MEDICAL/ETHICAL FRAMING (keep this in your report):
This models a SYMPTOM-AWARENESS / "when to see a doctor" triage tool,
NOT a diagnostic tool. Real cancer diagnosis requires imaging, biopsy,
or lab work -- no symptom checklist can confirm cancer. Many cancers
are also asymptomatic in early stages; this dataset only models
patients who are already symptomatic. A "No Red Flags" class is
included specifically so the model isn't forced to diagnose every
symptom combination as cancer.

Symptom probabilities per condition were derived from published
symptom descriptions (Mayo Clinic, American Cancer Society, Cleveland
Clinic). These are reasoned estimates translated from qualitative
descriptions ("hallmark", "advanced-stage only", "sometimes") into
probabilities -- NOT clinical measurements. Document this translation
and your reasoning in your report.

Usage:
    python generate_cancer_dataset.py

Produces:
    cancer_symptom_dataset.csv
"""

# ---------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------
# 2. Shared symptom vocabulary, grouped by body system for clarity
# ---------------------------------------------------------------
GENERAL = ["unexplained_weight_loss", "fatigue", "night_sweats", "persistent_pain"]
BREAST = ["breast_lump", "breast_skin_changes", "nipple_discharge",
          "nipple_inversion", "breast_swelling"]
LUNG = ["persistent_cough", "coughing_blood", "chest_pain",
        "shortness_of_breath", "hoarseness"]
CERVICAL = ["abnormal_vaginal_bleeding", "unusual_vaginal_discharge", "pelvic_pain"]

SYMPTOMS = GENERAL + BREAST + LUNG + CERVICAL   # 17 symptoms total

# Biological constraint -- forced to 0 for male patients regardless of
# sampled probability or noise (cervical cancer requires a cervix).
FEMALE_ONLY = ["abnormal_vaginal_bleeding", "unusual_vaginal_discharge"]


def make_profile(overrides: dict, default: float = 0.02) -> dict:
    """Start every symptom at a low background rate, then override
    with the disease's actual hallmark/associated symptoms."""
    profile = {s: default for s in SYMPTOMS}
    profile.update(overrides)
    return profile


# ---------------------------------------------------------------
# 3. Condition profiles: symptom probabilities + realistic sex mix
# ---------------------------------------------------------------
DISEASE_PROFILES = {
    "Breast Cancer": {
        "sex_distribution": {"F": 0.99, "M": 0.01},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.15, "fatigue": 0.25, "night_sweats": 0.10,
            "persistent_pain": 0.20, "breast_lump": 0.85, "breast_skin_changes": 0.35,
            "nipple_discharge": 0.20, "nipple_inversion": 0.15, "breast_swelling": 0.30,
        }),
    },
    "Lung Cancer": {
        "sex_distribution": {"F": 0.45, "M": 0.55},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.40, "fatigue": 0.35, "night_sweats": 0.15,
            "persistent_pain": 0.20, "persistent_cough": 0.75, "coughing_blood": 0.35,
            "chest_pain": 0.40, "shortness_of_breath": 0.45, "hoarseness": 0.25,
        }),
    },
    "Cervical Cancer": {
        "sex_distribution": {"F": 1.0},
        "symptoms": make_profile({
            "unexplained_weight_loss": 0.15, "fatigue": 0.25, "night_sweats": 0.08,
            "persistent_pain": 0.15, "abnormal_vaginal_bleeding": 0.75,
            "unusual_vaginal_discharge": 0.55, "pelvic_pain": 0.50,
        }),
    },
    "No Red Flags (Routine/Benign)": {
        "sex_distribution": {"F": 0.50, "M": 0.50},
        "symptoms": make_profile({
            "fatigue": 0.30, "persistent_pain": 0.10,
            "unexplained_weight_loss": 0.03, "night_sweats": 0.05,
        }, default=0.015),
    },
}

# ---------------------------------------------------------------
# 3b. No Red Flags controlled-ablation sub-patterns
# ---------------------------------------------------------------
NO_RED_FLAGS_CLASS = "No Red Flags (Routine/Benign)"
GENERIC_NO_RED_FLAGS_PROFILE = DISEASE_PROFILES[NO_RED_FLAGS_CLASS]


def make_no_red_flags_variant(overrides: dict) -> dict:
    """Copy the current benign profile and change only selected symptoms."""
    variant = {
        "sex_distribution": GENERIC_NO_RED_FLAGS_PROFILE["sex_distribution"].copy(),
        "symptoms": GENERIC_NO_RED_FLAGS_PROFILE["symptoms"].copy(),
    }
    variant["symptoms"].update(overrides)
    return variant


# The generic-mild group preserves the original benign generation logic. The
# breast look-alike models a benign lump without the rarer accompanying breast
# warning signs, testing whether the model over-relies on breast_lump alone.
# The gynecological look-alike models common benign bleeding and pelvic pain
# while keeping unusual discharge at its low background rate, testing whether
# the model over-relies on those overlapping symptoms.
NO_RED_FLAGS_SUBPATTERN_WEIGHTS = [0.70, 0.15, 0.15]
NO_RED_FLAGS_SUBPATTERN_NAMES = [
    "generic_mild",
    "breast_lookalike",
    "gyn_lookalike",
]
NO_RED_FLAGS_SUBPATTERN_PROFILES = [
    GENERIC_NO_RED_FLAGS_PROFILE,
    make_no_red_flags_variant({"breast_lump": 0.40}),
    make_no_red_flags_variant({
        "abnormal_vaginal_bleeding": 0.35,
        "pelvic_pain": 0.35,
    }),
]

# ---------------------------------------------------------------
# 4. Generation settings
# ---------------------------------------------------------------
SAMPLES_PER_CLASS = 300   # -> 1,200 total rows (balanced classes)
NOISE_PROB = 0.03         # small chance any given symptom randomly flips,
                           # modeling atypical presentation / comorbidities


def generate_patient(disease: str, profile: dict, patient_id: int) -> dict:
    sexes = list(profile["sex_distribution"].keys())
    probs = list(profile["sex_distribution"].values())
    sex = rng.choice(sexes, p=probs)
    age = int(rng.integers(20, 90))

    row = {"patient_id": patient_id, "age": age, "sex": sex}
    for symptom in SYMPTOMS:
        p = profile["symptoms"][symptom]
        present = rng.random() < p
        if rng.random() < NOISE_PROB:
            present = not present
        if symptom in FEMALE_ONLY and sex != "F":   # enforce biological plausibility
            present = False
        row[symptom] = int(present)
    row["diagnosis"] = disease
    return row


def main():
    rows = []
    patient_id = 1
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(SAMPLES_PER_CLASS):
            selected_profile = profile
            subpattern = "n/a"
            if disease == NO_RED_FLAGS_CLASS:
                subpattern_index = rng.choice(
                    len(NO_RED_FLAGS_SUBPATTERN_PROFILES),
                    p=NO_RED_FLAGS_SUBPATTERN_WEIGHTS,
                )
                selected_profile = NO_RED_FLAGS_SUBPATTERN_PROFILES[subpattern_index]
                subpattern = NO_RED_FLAGS_SUBPATTERN_NAMES[subpattern_index]

            row = generate_patient(disease, selected_profile, patient_id)
            row["subpattern"] = subpattern
            rows.append(row)
            patient_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    out_path = "cancer_symptom_dataset_v2.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass balance sanity check:")
    class_counts = df["diagnosis"].value_counts().sort_index()
    class_percentages = class_counts / len(df)
    class_balance = pd.DataFrame({
        "Count": class_counts,
        "Percentage": class_percentages,
    })
    print(class_balance)
    expected_class_percentage = 1 / len(class_counts)
    is_balanced = np.allclose(class_percentages, expected_class_percentage)
    print(
        f"All classes balanced at {expected_class_percentage:.2%}: "
        f"{is_balanced}"
    )

    no_red_flags = df[df["diagnosis"] == NO_RED_FLAGS_CLASS]
    print("\nNo Red Flags look-alike symptom prevalence:")
    print(f"breast_lump=1: {no_red_flags['breast_lump'].mean():.2%}")
    print(
        f"abnormal_vaginal_bleeding=1: "
        f"{no_red_flags['abnormal_vaginal_bleeding'].mean():.2%}"
    )
    print("\nNo Red Flags sub-pattern counts:")
    subpattern_counts = no_red_flags["subpattern"].value_counts().reindex(
        NO_RED_FLAGS_SUBPATTERN_NAMES,
        fill_value=0,
    )
    print(subpattern_counts)
    print("\nSex distribution by diagnosis (sanity check for biological plausibility):")
    print(pd.crosstab(df["diagnosis"], df["sex"]))
    print("\nHallmark symptom prevalence by diagnosis (sanity check):")
    hallmark_cols = ["breast_lump", "persistent_cough", "abnormal_vaginal_bleeding"]
    print(df.groupby("diagnosis")[hallmark_cols].mean().round(2))


if __name__ == "__main__":
    main()

# %% [markdown]
# ## 5. Original vs. v2 Ablation Comparison
#
# This final section reruns the original accuracy-based tuning on both datasets,
# compares metrics and confusion matrices, and analyzes No Red Flags errors by
# v2 subpattern. It intentionally does not apply the Phase 2b cost-sensitive
# scorer.

# %% [markdown]
# # Phase 2 Ablation Comparison: Original Dataset vs. Dataset v2
#
# This comparison mirrors the existing Phase 1 preprocessing and Phase 2
# accuracy-based GridSearchCV. It intentionally does not use the Phase 2b
# cost-sensitive scorer.

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

RANDOM_STATE = 42
DATASET_PATHS = {
    "Original dataset": "cancer_symptom_dataset.csv",
    "v2 dataset": "cancer_symptom_dataset_v2.csv",
}

# This is the original Phase 2 search space, with no cost-sensitive scorer.
PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8, 10, None],
    "min_samples_leaf": [1, 2, 5, 10, 15, 20],
    "criterion": ["gini", "entropy"],
}


def run_pipeline(dataset_path: str) -> dict:
    """Run the same preprocessing, split, tuning, and evaluation on one CSV."""
    df = pd.read_csv(dataset_path)
    df_clean = df.drop(columns=["patient_id"])
    subpattern = None
    if "subpattern" in df_clean.columns:
        # This is an analysis-only label. Preserve it for subgroup analysis,
        # but explicitly remove it before constructing the model feature set.
        subpattern = df_clean["subpattern"].copy()
        df_clean = df_clean.drop(columns=["subpattern"])
    df_clean["sex_encoded"] = df_clean["sex"].map({"M": 0, "F": 1})

    symptom_cols = [
        col
        for col in df_clean.columns
        if col not in ["age", "sex", "diagnosis", "sex_encoded"]
    ]
    feature_cols = symptom_cols + ["age", "sex_encoded"]
    X = df_clean[feature_cols]
    y = df_clean["diagnosis"]

    if subpattern is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        subpattern_test = None
    else:
        X_train, X_test, y_train, y_test, _, subpattern_test = train_test_split(
            X,
            y,
            subpattern,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    classes = model.classes_

    return {
        "best_params": grid_search.best_params_,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "y_test": y_test,
        "y_test_pred": pd.Series(y_test_pred, index=y_test.index),
        "subpattern_test": subpattern_test,
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, y_test_pred, labels=classes),
            index=classes,
            columns=classes,
        ),
        "classes": classes,
    }


def print_comparison(results: dict) -> None:
    original = results["Original dataset"]
    v2 = results["v2 dataset"]

    overall_table = pd.DataFrame(
        {
            "Original dataset": {
                "Test accuracy": original["test_accuracy"],
                "Train accuracy": original["train_accuracy"],
                "Train/test gap (percentage points)": (
                    original["train_accuracy"] - original["test_accuracy"]
                )
                * 100,
            },
            "v2 dataset": {
                "Test accuracy": v2["test_accuracy"],
                "Train accuracy": v2["train_accuracy"],
                "Train/test gap (percentage points)": (
                    v2["train_accuracy"] - v2["test_accuracy"]
                )
                * 100,
            },
        }
    )
    print("\n=== Overall Accuracy Comparison ===")
    print(overall_table.to_string(float_format=lambda value: f"{value:.4f}"))

    metric_rows = []
    metric_names = ["precision", "recall", "f1-score"]
    for class_name in original["classes"]:
        for metric_name in metric_names:
            metric_rows.append(
                {
                    "Class": class_name,
                    "Metric": metric_name,
                    "Original dataset": original["classification_report"][class_name][
                        metric_name
                    ],
                    "v2 dataset": v2["classification_report"][class_name][
                        metric_name
                    ],
                }
            )

    print("\n=== Per-Class Precision / Recall / F1 Comparison ===")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    confusion_matrices = pd.concat(
        {
            "Original dataset": original["confusion_matrix"],
            "v2 dataset": v2["confusion_matrix"],
        },
        axis=1,
    )
    print("\n=== Confusion Matrix Comparison ===")
    print("Rows are actual classes; columns are predicted classes.")
    print(confusion_matrices)

    # The comparison tolerance is a reporting convention for calling a metric
    # "roughly unchanged," not part of model training or selection.
    roughly_unchanged_tolerance = 0.05
    class_metrics = {
        class_name: {
            metric_name: (
                original["classification_report"][class_name][metric_name],
                v2["classification_report"][class_name][metric_name],
            )
            for metric_name in ["precision", "recall"]
        }
        for class_name in original["classes"]
    }

    def metric_change(class_name: str, metric_name: str) -> float:
        original_value, v2_value = class_metrics[class_name][metric_name]
        return v2_value - original_value

    print("\n=== Plain-Language Ablation Interpretation ===")
    for class_name in ["Breast Cancer", "Cervical Cancer"]:
        precision_change = metric_change(class_name, "precision")
        recall_change = metric_change(class_name, "recall")
        precision_direction = "dropped" if precision_change < 0 else "did not drop"
        recall_direction = "dropped" if recall_change < 0 else "did not drop"
        print(
            f"{class_name}: precision {precision_direction} "
            f"({precision_change:+.4f}); recall {recall_direction} "
            f"({recall_change:+.4f})."
        )

    lung_precision_change = metric_change("Lung Cancer", "precision")
    lung_recall_change = metric_change("Lung Cancer", "recall")
    lung_unchanged = (
        abs(lung_precision_change) <= roughly_unchanged_tolerance
        and abs(lung_recall_change) <= roughly_unchanged_tolerance
    )
    lung_summary = "stayed roughly unchanged" if lung_unchanged else "changed materially"
    print(
        f"Lung Cancer {lung_summary}: precision change "
        f"{lung_precision_change:+.4f}; recall change "
        f"{lung_recall_change:+.4f}. Its profile was not modified."
    )
    print(
        "This does not make the dataset uniformly harder: it specifically tests "
        "robustness where the original model could rely on a single near-perfect "
        "signature symptom, while leaving the Lung Cancer profile untouched."
    )


def print_v2_subpattern_breakdown(v2_result: dict) -> None:
    """Measure No Red Flags errors by the analysis-only v2 sub-pattern label."""
    if v2_result["subpattern_test"] is None:
        raise ValueError("The v2 comparison requires the 'subpattern' column.")

    test_analysis = pd.DataFrame(
        {
            "true_diagnosis": v2_result["y_test"],
            "predicted_diagnosis": v2_result["y_test_pred"],
            "subpattern": v2_result["subpattern_test"],
        }
    )
    no_red_flags_test = test_analysis[
        test_analysis["true_diagnosis"] == "No Red Flags (Routine/Benign)"
    ]

    subpattern_names = ["generic_mild", "breast_lookalike", "gyn_lookalike"]
    breakdown_rows = []
    for subpattern_name in subpattern_names:
        subpattern_test = no_red_flags_test[
            no_red_flags_test["subpattern"] == subpattern_name
        ]
        misclassified = subpattern_test[
            subpattern_test["predicted_diagnosis"]
            != "No Red Flags (Routine/Benign)"
        ]
        misclassified_as_counts = misclassified["predicted_diagnosis"].value_counts().sort_index()
        misclassified_as = ", ".join(
            f"{class_name} ({count})"
            for class_name, count in misclassified_as_counts.items()
        )

        breakdown_rows.append(
            {
                "sub-pattern": subpattern_name,
                "n": len(subpattern_test),
                "correct": len(subpattern_test) - len(misclassified),
                "misclassified": len(misclassified),
                "misclassified-as": misclassified_as or "None",
            }
        )

    breakdown_table = pd.DataFrame(breakdown_rows)
    print("\n=== v2 No Red Flags Sub-Pattern Breakdown ===")
    print(f"No Red Flags patients in the test set: {len(no_red_flags_test)}")
    print(breakdown_table.to_string(index=False))

    generic_row = breakdown_table[
        breakdown_table["sub-pattern"] == "generic_mild"
    ].iloc[0]
    lookalike_rows = breakdown_table[
        breakdown_table["sub-pattern"].isin(
            ["breast_lookalike", "gyn_lookalike"]
        )
    ]
    generic_rate = generic_row["misclassified"] / generic_row["n"]
    lookalike_n = lookalike_rows["n"].sum()
    lookalike_misclassified = lookalike_rows["misclassified"].sum()
    lookalike_rate = lookalike_misclassified / lookalike_n
    rate_difference = lookalike_rate - generic_rate

    # Treat differences within five percentage points as similar so the
    # interpretation does not overstate a small subgroup fluctuation.
    rate_similarity_tolerance = 0.05
    if abs(rate_difference) <= rate_similarity_tolerance:
        rate_comparison = "similar to"
        conclusion = "The look-alike groups do not show a meaningfully higher rate."
    elif rate_difference > 0:
        rate_comparison = "higher than"
        conclusion = "The look-alike groups have the higher rate, as predicted."
    else:
        rate_comparison = "lower than"
        conclusion = "The look-alike groups do not have the higher rate predicted."

    print("\n=== Plain-Language Sub-Pattern Interpretation ===")
    print(
        f"generic_mild misclassification rate: "
        f"{generic_row['misclassified']}/{generic_row['n']} "
        f"({generic_rate:.2%}); combined look-alike rate: "
        f"{lookalike_misclassified}/{lookalike_n} ({lookalike_rate:.2%})."
    )
    print(
        f"The combined look-alike rate is {rate_comparison} the generic_mild "
        f"rate (difference {rate_difference:+.2%}). {conclusion}"
    )


if __name__ == "__main__":
    results = {
        dataset_name: run_pipeline(dataset_path)
        for dataset_name, dataset_path in DATASET_PATHS.items()
    }

    print("\n=== Best Parameters ===")
    for dataset_name, result in results.items():
        print(f"{dataset_name}: {result['best_params']}")

    print_comparison(results)
    print_v2_subpattern_breakdown(results["v2 dataset"])
