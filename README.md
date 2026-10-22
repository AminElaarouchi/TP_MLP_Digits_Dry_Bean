# TP_MLP_Digits_Dry_Bean

This practical assignment explores multilayer perceptrons (MLPs) through a NumPy XNOR example, handwritten digit recognition, and Dry Bean variety classification.

## Notebook

`TP_MLP_Digits_Dry_Bean(1).ipynb`

The notebook includes explanations, exercises, answers to questions Q1–Q14, model comparisons, and a final classification experiment.

## Objectives

- Build networks with Keras `Sequential` and `Dense`.
- Configure training with `compile` and `fit`.
- Generate predictions and evaluate classification performance.
- Compare learning rates, optimizers, batch sizes, and architectures.
- Explore L1/L2, Dropout, Batch Normalization, and early stopping.
- Save and reload models and preprocessing objects.

## Datasets

| Dataset | Features | Classes | Source |
| --- | --- | --- | --- |
| Digits | 64 pixel intensities from 8 × 8 images; 1,797 observations | 10 digits | Built into scikit-learn |
| Dry Bean | 16 numerical size and shape measurements | 7 bean varieties | UCI via `ucimlrepo` |

Dry Bean uses extracted measurements, not raw photographs. Exact duplicate rows are removed before splitting.

Source: [Dry Bean — UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)  
DOI: `10.24432/C50S4B` — License: CC BY 4.0.

## Installation

The notebook runs in Google Colab or local Jupyter. CPU execution is sufficient.

Install dependencies in a notebook cell:

```python
%pip install tensorflow numpy matplotlib scikit-learn pandas ucimlrepo joblib
```

Restart the kernel if necessary.

For local Jupyter, install and launch JupyterLab:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

Open the notebook using the environment containing the dependencies.

Internet access is required to retrieve Dry Bean. Digits requires no additional dataset download after installing scikit-learn.

## Running the TP

1. Open the notebook in Jupyter or Colab.
2. Run the cells in order, starting with installation and imports.
3. Inspect the NumPy XNOR example.
4. Execute the Digits training and comparison sections.
5. Run the Dry Bean section and final experiment.
6. Review the generated metrics, plots, answers, and exports.
7. Save the notebook with its outputs.

The Dry Bean section redefines its imports and helper functions and can run in a fresh kernel from its own setup cells.

Some experiments, particularly batch size 1, can take several minutes. Training uses `verbose=0`, so progress logs are suppressed.

## Methodology

Both datasets use a stratified split of approximately:

- 60% training
- 20% validation
- 20% test

`StandardScaler` is fitted only on training data. The same fitted scaler transforms validation and test data.

Dry Bean class names are converted to integer labels using `LabelEncoder`.

### Baseline models

| Dataset | Architecture | Parameters |
| --- | --- | --- |
| Digits | 64 → 32 ReLU → 10 softmax | 2,410 |
| Dry Bean | 16 → 32 ReLU → 7 softmax | 775 |

Initial training uses Adam at learning rate `0.001`, sparse categorical cross-entropy, 40 epochs, and batches of 32.

### Comparisons

| Factor | Configurations |
| --- | --- |
| Learning rate | `0.0001`, `0.001`, `0.05` |
| Optimizer | Adam, SGD |
| Batch size | 1, 32, full training set |
| Hidden layers | `(2,)`, `(32,)`, `(64, 32)` |
| Activation | ReLU, tanh |
| Regularization/normalization | L1, L2, Dropout, BatchNorm |

Early stopping monitors `val_loss`, uses `patience=8`, restores the best weights, and allows a maximum of 100 epochs.

With L1/L2, the loss returned by `evaluate` includes the regularization penalty. The notebook also compares cross-entropy without this penalty.

## Dry Bean Final Experiment

Two models with hidden layers `(64, 32)` are compared:

- Without Dropout
- With Dropout `0.25`

The model is selected by highest validation macro F1, with lowest validation cross-entropy as the tie-breaker.

The selected model is then evaluated on the reserved test set and compared with a majority-class baseline.

Evaluation includes:

- Accuracy and macro F1
- Per-class precision, recall, and F1
- Confusion matrix
- Examples of incorrect predictions
- A generated conclusion

The Digits test set remains reserved in the demonstration.

## Generated Files

Files are saved in the kernel's current working directory.

| File | Purpose |
| --- | --- |
| `modele_mlp.keras` | Digits model |
| `modele_mlp_dry_bean_demo.keras` | Dry Bean demonstration model |
| `modele_mlp_dry_bean.keras` | Selected final Dry Bean model |
| `pretraitement_dry_bean.joblib` | Scaler, label encoder, and ordered feature names |
| `metadata_dry_bean.json` | Configuration, classes, versions, and evaluation metrics |

A `.keras` file does not contain the scikit-learn preprocessing.

For new Dry Bean observations, preserve the saved feature order, apply the saved scaler's `transform`, call `predict`, and decode predicted indices with the saved encoder. Do not refit preprocessing on new observations.

Rerunning export cells overwrites files with the same names. Download exported files before ending a Colab session.

## Execution Status and Reproducibility

The supplied Digits section contains saved outputs. The Dry Bean section is implemented but its training results still need to be generated by executing the notebook.

The notebook uses `SEED = 42`. Results may vary with hardware, package versions, and numerical operations.

The recorded Digits environment reports TensorFlow `2.21.0` and Keras `3.15.1`; dependencies are not pinned.

## Submission Checklist

- Run all required cells and retain their outputs.
- Complete questions Q1–Q14 and the Dry Bean final experiment.
- Interpret the results produced by your execution.
- Use validation data for model selection.
- Keep settings fixed after final test evaluation.
- Confirm that model save/reload checks pass.
