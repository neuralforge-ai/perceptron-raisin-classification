# Single-Layer Perceptron from Scratch: Raisin Variety Classification

A binary classifier built **from scratch using only Python and NumPy** (no scikit-learn or other prebuilt ML models) that predicts whether a raisin is of the **Kecimen** or **Besni** variety from its shape measurements.

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Perceptron Algorithm](#perceptron-algorithm)
- [Results](#results)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Limitations and Observations](#limitations-and-observations)
- [Citation](#citation)

## Dataset

| Property | Details |
|---|---|
| **Name** | Raisin Dataset |
| **Source** | UCI Machine Learning Repository |
| **Link** | https://archive.ics.uci.edu/dataset/850/raisin |
| **Samples** | 900 (450 Kecimen, 450 Besni) |
| **Features** | 7 numeric morphological features |
| **Task** | Binary classification |

**Features:** Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter.

**Target:** `Class` (Kecimen = 0, Besni = 1).

The features were extracted from images of raisins grown in Turkey. The dataset is loaded directly in the notebook with the `ucimlrepo` package (dataset ID 850).

## Project Workflow

1. **Load data** from the UCI repository.
2. **Preprocess:**
   - Check for missing values.
   - Encode labels (Kecimen = 0, Besni = 1).
   - Shuffle the data and split 80% train / 20% test, using NumPy.
   - Standardize features (zero mean, unit variance) using **training-set statistics only**, to avoid data leakage.
3. **Implement** the Single-Layer Perceptron from scratch.
4. **Train** the model and plot misclassifications per epoch.
5. **Evaluate** on the test set using metrics computed from scratch.
6. **Visualize** the confusion matrix.

## Perceptron Algorithm

For each training sample `x` with target `y`:

```
z   = w · x + b
ŷ   = 1 if z >= 0 else 0        (step activation)
w  ← w + η (y − ŷ) x
b  ← b + η (y − ŷ)
```

where `η` is the learning rate. Weights are initialized to zero and updated sample by sample (online learning) for a fixed number of epochs.

**Hyperparameters used:** learning rate = `0.01`, epochs = `100`.

## Results

> Replace the placeholders below with the actual values from your notebook output.

| Metric | Value |
|---|---|
| Accuracy | `XX.XX%` |
| Precision | `X.XXXX` |
| Recall | `X.XXXX` |
| F1-Score | `X.XXXX` |

**Confusion matrix (test set):**

|  | Predicted Kecimen | Predicted Besni |
|---|---|---|
| **Actual Kecimen** | TN | FP |
| **Actual Besni** | FN | TP |

Add your screenshots here:

```markdown
![Training Error](images/training_error.png)
![Confusion Matrix](images/confusion_matrix.png)
```

## How to Run

### Option 1: Google Colab (recommended)
1. Open `Perceptron_Raisin.ipynb` in [Google Colab](https://colab.research.google.com).
2. Click **Runtime → Run all**.

### Option 2: Locally
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install numpy pandas matplotlib ucimlrepo
jupyter notebook Perceptron_Raisin.ipynb
```

**Requirements:** Python 3.8+, NumPy, pandas (data loading only), Matplotlib, ucimlrepo.

> pandas is used only to read the dataset. The model, train/test split, scaling and all metrics are implemented with NumPy.

## Repository Structure

```
.
├── Perceptron_Raisin.ipynb   # Complete code with outputs
├── perceptron_raisin.py      # Same code as a Python script (optional)
├── images/                   # Output plots (optional)
└── README.md
```

## Limitations and Observations

- A single-layer perceptron can only learn a **linear decision boundary**, and it converges only if the classes are perfectly linearly separable.
- The two raisin varieties overlap in feature space, so the training error **fluctuates across epochs** rather than reaching zero.
- Possible improvements: learning-rate decay, averaged perceptron, feature selection, or a multi-layer network.

## Citation

Çinar, İ., Koklu, M., & Tasdemir, S. (2020). *Raisin* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5660T

## Author

**Amna Khalid**
BS-AI 
