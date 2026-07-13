# SMILES-2026 Spatial_KMM
Used environment:
Python 3.12.6
numpy 2.4.2
matplotlib 3.10.0
scikit-learn 1.6.1
scipy 1.17.1

The main solution is in the file `KMM.ipynb`. It used `Jupyter Notebook` to run the `KMM.ipynb`, the main functions for tasks are in the file `KMM_functions.py`, 
the methods realization are in the `Metthods.py`. 
To run the project: Run all cells of the `KMM.ipynb`.
Runtime: ~27 sec.

## Literature 

## Tasks of the project
1. Formalize the estimand for KMM-weighted spatial cross-validation.
2. Implement global and fold-specific KMM weighting for validation losses.
3. Compare random CV, spatial CV, importance-weighted CV, and KMM-weighted spatial CV on controlled spatial experiments.
4. Report diagnostics for effective sample size, weight concentration, support overlap, and sensitivity to kernel bandwidth and block design.
## Methods
`Random CV`

`Spatial CV`

`Importance-weighted CV`

`KMM-weighted spatial CV`

## Metrics
• *Risk-estimation bias*: difference between estimated CV error and known deployment error in simulations.
• *Absolute and relative bias*: $|\{hat}R_{CV} - R_{dep}|$ and the same error normalized by $R_{dep}$.
• *Variance and stability*: variability across repeated fold partitions and KMM hyperparameters.
• *Overlap quality*: effective sample size, maximum weight, clipped-weight share, and MMD before and after weighting.
• *Model-selection impact*: whether weighted spatial CV selects hyperparameters that improve deployment error.
• *Failure detection*: whether low effective sample size, high clipping rate, or residual MMD predict cases where the
weighted estimate is unreliable.

## Results