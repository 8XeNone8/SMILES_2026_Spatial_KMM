# SMILES_2026_Spatial_KMM
This repository is for SMILES 2026 student competition as part of the implementation of projects. Here we present the abstract of the project. 
## Abstract 
The project is dedicated to research in the field of ***Geospatial machine-learning systems(GMLS)***. ***GMLS*** used for several tasks, for example, environmental monitoring, ecosystem
assessment and risk forecasting.  
There are three global problems with *geospatial data*: spatial autocorrelation, uneven sampling and distribution shift between sampled and target regions.
The crusial problem is to evaluate how accurate predictions will perform at future deployment locations.
We present a comparison of models' forecast estimates of the methods `random cross-validation(CV)`, `spatial CV`, `importance-weighted CV` and `Kernel Mean Matching-weighted spatial CV`.
We tested CV methods on the `linear`, `exponential`, `periodic`, `random` and `random_autocorr` syntetic data frames with simple MSE metrics. We defined that the highest MSE value corresponds to the *importance-weighted CV*, which indicates the instability of the method.

Next tasks are: 1. used other metrics, realize Fold KMM with global weight and random, formalize the estimand for KMM-weighted spatial cross-validation with shift and bias.

 

