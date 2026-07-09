import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def kmm_weights(X_train, X_test, gamma=1.0, B=10):
    n_train = X_train.shape[0]
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    kappa = np.mean(rbf_kernel(X_train, X_test, gamma=gamma), axis=1)
    
    def objective(w):
        return 0.5 * w @ K @ w - kappa @ w

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - n_train}]
    bounds = [(0, B)] * n_train
    initial = np.ones(n_train)  

    res = minimize(objective, initial, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-8})
    if not res.success:
        print("KMM не сошёлся, используются равные веса.")
        return np.ones(n_train)
    return res.x

def spatial_kmm_cv(X, y, n_folds=5, gamma=1.0, B=10):
    kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    fold_ids = kmeans.fit_predict(X)

    errors_kmm = []
    errors_unweighted = []

    for test_fold in range(n_folds):
        test_idx = np.where(fold_ids == test_fold)[0]
        train_idx = np.where(fold_ids != test_fold)[0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        weights = kmm_weights(X_train, X_test, gamma=gamma, B=B)

        model_w = LinearRegression()
        model_w.fit(X_train, y_train, sample_weight=weights)
        y_pred_w = model_w.predict(X_test)
        error_kmm = mean_squared_error(y_test, y_pred_w)

        model_uw = LinearRegression()
        model_uw.fit(X_train, y_train)
        y_pred_uw = model_uw.predict(X_test)
        error_unw = mean_squared_error(y_test, y_pred_uw)

        errors_kmm.append(error_kmm)
        errors_unweighted.append(error_unw)

    return np.mean(errors_kmm), np.mean(errors_unweighted)