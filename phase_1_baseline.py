# %% [markdown]
# # Phase 1: EDA, Preprocessing, and Baseline Decision Tree Classifier
# **Course:** APT3010A Intro to AI - Cancer Symptom Screener Project
# 
# This file serves as the pipeline for Phase 1. It is formatted with `# %%` cell delimiters 
# so it can be run either as a standard Python script or as interactive notebook cells in VS Code/Google Colab.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
print()

# C. Crosstab of Diagnosis by Sex
# RATIONALE:
# We cross-tabulate diagnosis against sex. Medically, certain cancers have strict biological links:
# for instance, cervical cancer should only occur in biological females (F) in the dataset.
# Verifying this distribution ensures there are no demographic anomalies in our data generator.
print("--- Crosstab of Diagnosis by Sex ---")
crosstab_sex = pd.crosstab(df_clean["diagnosis"], df_clean["sex"])
print(crosstab_sex)
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
# information gain (Gini impurity decrease) is deterministic and reproducible.
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Test Accuracy: {accuracy * 100:.2f}%")

# Generate and print the confusion matrix
classes = clf.classes_
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

print("\n--- Confusion Matrix ---")
print(cm_df)

# Print a detailed classification report containing precision, recall, and F1-score per class
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=classes))
