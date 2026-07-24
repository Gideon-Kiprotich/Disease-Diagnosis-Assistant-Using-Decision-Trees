# %% [markdown]
# # Phase 2: Hyperparameter Tuning, Evaluation, and Patient Walkthroughs
# **Course:** APT3010A Intro to AI - Cancer Symptom Screener Project
# 
# This file completes Phase 2 of the pipeline. It is formatted with `# %%` cell delimiters 
# so it can be run either as a standard Python script or as interactive notebook cells in VS Code/Google Colab.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Setup: Load Dataset and Reproduce Phase 1 Train/Test Split
# RATIONALE: We use the exact same data cleaning, encoding (`sex_encoded`), and stratified 80/20 train/test split (`random_state=42`) from Phase 1 to ensure direct comparability across model versions.

# %%
print("=== Phase 2 Setup: Loading Data & Train/Test Split ===")
df = pd.read_csv("cancer_symptom_dataset.csv")
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
    
    # Extract decision path
    node_indicator = clf_best.decision_path(patient_row)
    leaf_id = clf_best.apply(patient_row)[0]
    node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    tree_ = clf_best.tree_
    print("\nDecision Tree Path Traversal:")
    
    path_features = []
    for step_num, node_id in enumerate(node_indices, 1):
        if node_id == leaf_id:
            pred_class_idx = np.argmax(tree_.value[node_id])
            pred_class_name = classes[pred_class_idx]
            samples_in_leaf = tree_.n_node_samples[node_id]
            print(f"  Step {step_num} [LEAF NODE {node_id}]: Reached terminal node with {samples_in_leaf} training samples. Final Prediction -> '{pred_class_name}'")
        else:
            feat_idx = tree_.feature[node_id]
            feat_name = feature_cols[feat_idx]
            path_features.append(feat_name)
            threshold = tree_.threshold[node_id]
            val = patient_data[feat_name]
            
            if val <= threshold:
                decision_str = f"<= {threshold:.2f} (Rule satisfied: True -> Go Left)"
            else:
                decision_str = f"> {threshold:.2f} (Rule satisfied: False -> Go Right)"
                
            print(f"  Step {step_num} [Node {node_id}]: Split on '{feat_name}' | Patient value: {val} {decision_str}")

    # Dynamic clinical rationale generated from actual traced path features
    if path_features:
        if len(path_features) == 1:
            features_str = f"checking '{path_features[0]}'"
        elif len(path_features) == 2:
            features_str = f"first checking '{path_features[0]}', then '{path_features[1]}'"
        else:
            features_str = f"first checking '{path_features[0]}', then " + ", then ".join([f"'{f}'" for f in path_features[1:]])
        print(f"\nClinical Rationale: The tree reached this diagnosis by {features_str}.")

print("\n" + "="*50 + "\n")
