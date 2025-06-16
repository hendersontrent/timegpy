import numpy as np
import pandas as pd
import pingouin as pg

def compute_eta_squared(feature_vector, y):
    """
    Computes partial eta squared using Pingouin's ANOVA implementation.
    """

    # Ensure input is 1D
    feature_vector = np.ravel(feature_vector)

    # Remove NaNs and non-finite values
    finite_mask = np.isfinite(feature_vector)
    X_clean = feature_vector[finite_mask]
    y_clean = y[finite_mask]

    if len(np.unique(y_clean)) < 2 or len(y_clean) < 3:
        return 0.0

    df = pd.DataFrame({'feature': X_clean, 'group': y_clean})

    try:
        aov = pg.anova(data=df, dv='feature', between='group', detailed=True)
        eta_squared = aov['np2'][0]
    except Exception:
        eta_squared = 0.0

    return eta_squared
