# Wearable Exam QLSTM

**Edge-Ready Multimodal Wearable Intelligence for Predicting Exam Performance**  
*A QLSTM study on real-world cognitive stress data*

This repository provides a reproducible pipeline for exploring and modeling **real-world multimodal wearable signals** (EDA, HR, ACC, etc.) collected during academic exams, with the goal of predicting **exam performance (grades)** under cognitive stress.

The project is built on the **PhysioNet Wearable Exam Stress dataset**, emphasizing:
- real-world noise and motion artifacts,
- long-duration time-series,
- small-sample, subject-level evaluation (LOSO),
- and edge-ready modeling considerations.

---

## 📂 Repository Structure

```

wearable-exam-qlstm/
├── data/                      # Dataset directory (NOT tracked by git)
│   └── wearable-exam-stress/  # PhysioNet dataset root
├── src/
│   ├── data_loader.py         # Robust W1/W2 data loader (signals + grades)
│   ├── preprocess.py          # (W2) resampling / windowing (future)
│   ├── make_dataset.py        # (W2) dataset construction (future)
│   └── models/                # (W3+) baselines / QLSTM
├── plot_w1_signal.py          # W1: raw signal visualization script
├── figures/                   # Generated plots (ignored by git)
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 📊 Dataset

This project uses the **Wearable Exam Stress Dataset** hosted on PhysioNet.

- **Subjects**: 10 university students  
- **Exams**: Midterm 1, Midterm 2, Final  
- **Modalities**:
  - EDA (Electrodermal Activity)
  - HR (Heart Rate)
  - BVP
  - IBI
  - TEMP
  - ACC (3-axis acceleration)
- **Context**: Real exam sessions (9:00 AM start), containing motion artifacts and realistic noise
- **Labels**: Exam grades stored in `StudentGrades.txt`

Dataset page:  
👉 https://physionet.org/content/wearable-exam-stress/1.0.0/

---

## ⚖️ License & Data Usage

### Dataset License
The PhysioNet dataset is released under:

> **Open Data Commons Attribution License v1.0 (ODC-By)**

You are free to share and adapt the data **with proper attribution**.

### Citation Requirement
Any use of the dataset **must cite**:

- Amin et al., *Wearable Exam Stress Dataset*, PhysioNet, 2022  
- Goldberger et al., *PhysioNet: Components of a New Research Resource*, Circulation, 2000  

See **CITATION.cff** for BibTeX / citation metadata.

---

## ⬇️ Data Download

⚠️ **Raw data is NOT included in this repository.**

Download manually from PhysioNet:

```bash
mkdir -p data
cd data
wget -r -N -c -np https://physionet.org/files/wearable-exam-stress/1.0.0/
mv physionet.org/files/wearable-exam-stress/1.0.0 wearable-exam-stress
````

Final expected path:

```
data/wearable-exam-stress/
├── S1/
├── S2/
├── ...
├── S10/
└── StudentGrades.txt
```

---

## 🐍 Environment Setup

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

**Windows (Git Bash / PowerShell):**

```bash
.venv/Scripts/activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ✅ Week 1 (W1): Raw Signal Visualization & Grade Alignment

### Goal

* Verify data loading
* Visualize raw physiological signals
* Align exam sessions with grades
* Inspect real-world motion artifacts

### Run W1 script

```bash
python plot_w1_signal.py \
  --root data/wearable-exam-stress \
  --subject S1 \
  --exam "Midterm 1"
```

### Output

* A 3-panel plot:

  1. EDA
  2. HR
  3. ACC magnitude
* X-axis: minutes since exam start
* Title includes subject ID, exam name, and grade

Example:

```
figures/S1_midterm1_w1.png
```

### Notes

* ACC magnitude is **percentile-clipped** to remove parsing artifacts caused by timestamp rows in raw CSVs.
* Motion variability is intentionally preserved to reflect real-world conditions.

---

## 🔧 Data Loader Highlights (`src/data_loader.py`)

The data loader is designed to be **robust and research-grade**:

* Handles:

  * Non-UTF8 encodings (Windows-encoded grade files)
  * Irregular CSV formats (Empatica headers, missing timestamps)
  * Section-based grade files (MIDTERM 1 / MIDTERM 2 / FINAL)
* Normalizes subject IDs (`S01 → S1`)
* Provides reusable APIs for:

  * signal loading
  * grade parsing
  * ACC magnitude computation

This loader is reused in **W2 (windowing)** and **W3 (modeling)**.

---

## 🚧 Roadmap

* **W2**: Resampling, windowing, feature construction
* **W3**: Baseline models (Ridge, RF, LSTM, TCN)
* **W4**: QLSTM implementation
* **W5**: Motion artifact analysis & ablation
* **W6–W8**: Paper writing and submission

---

## 📖 Citation

If you use this code or dataset, please cite:

> Amin, M. R., et al. (2022).
> *A Wearable Exam Stress Dataset for Predicting Cognitive Performance.*
> PhysioNet. [https://doi.org/10.13026/kvkb-aj90](https://doi.org/10.13026/kvkb-aj90)

> Goldberger, A. L., et al. (2000).
> *PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.*
> Circulation.

---

## 👩‍💻 Maintainer

**Hsing-Tzu Ko (柯幸孜)**
Graduate Program in Smart Medicine & Health Informatics
National Taiwan University

---

> This project focuses on **real-world robustness**, **interpretability**, and **edge-ready intelligence** for wearable health analytics.

