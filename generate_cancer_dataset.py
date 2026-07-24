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

import numpy as np
import pandas as pd

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

    out_path = "cancer_symptom_dataset_regenerated.csv"
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