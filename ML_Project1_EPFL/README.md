# ML Project 1 — EPFL

Scaffold based on the supplied project description. All function bodies contain
`pass`; no algorithms, helper functions, or prediction pipeline are implemented.

```text
ML_Project1_EPFL/
├── implementations.py  # Six required methods and loss/gradient utilities
├── helpers.py          # Data loading, submission, batching and prediction stubs
├── preprocessing.py    # Training-fitted preprocessing and label encoding
├── validation.py       # Splits, cross-validation, accuracy and F1
├── run.py              # Training and prediction pipeline
├── requirements.txt    # NumPy; optional plotting dependencies listed as comments
├── README.md
├── report.tex          # Empty LaTeX report outline
├── report.pdf          # Placeholder; replace with the compiled final report
└── .gitignore
```

## Method contracts

The six required methods in `implementations.py` must return `(w, loss)`.
Use one-dimensional arrays for labels and weights. MSE includes the factor 0.5;
SGD uses one sample per step. Returned losses exclude regularization penalties.
Logistic methods require labels in `{0, 1}`. Least squares uses normal equations,
without `numpy.linalg.lstsq`.

Utility functions beyond those six methods are suggested scaffolding; their
interfaces may be adapted. The helpers are empty local stubs, not copies of the
official EPFL helpers. Check the official loader interface before implementing it.

## Next steps

1. Implement the methods and helpers.
2. Place `x_train.csv`, `y_train.csv`, and `x_test.csv` together in `data/`.
3. Fit preprocessing only on training partitions during local validation.
4. Choose and document hyperparameters, random seeds and the prediction threshold.
5. Complete `run.py` to reproduce the selected competition submission.
6. Complete `report.tex` and compile it into `report.pdf`.

The project permits the Python standard library and NumPy, with visualization
libraries for plotting only. Keep the report within two pages, with references
allowed on an additional third page.

After implementing the pipeline:

```bash
python -m pip install -r requirements.txt
python run.py
```
