import numpy as np
from sklearn.feature_selection import f_classif

def compute_eta_squared(feature_vector, y):
    
    # Ensure 2D array and remove NaNs
    X_col = feature_vector.reshape(-1, 1)
    finite_mask = np.isfinite(X_col.ravel())
    X_col = X_col[finite_mask].reshape(-1, 1)
    y_clean = y[finite_mask]

    if len(np.unique(y_clean)) < 2 or len(y_clean) < 3:
        return 0.0

    F, _ = f_classif(X_col, y_clean)
    eta_squared = F[0] / (F[0] + len(y_clean) - 2) if len(F) > 0 else 0.0
    return eta_squared