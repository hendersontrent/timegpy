import numpy as np
import re
from scipy.stats import zscore

def evaluate_expression(expr, X, z_score=True):
    """
    Evaluates a timegpy-format string expression of a time-average feature (like "mean((sin(X_t+0) * X_t+2))")
    over each row (time series) of the matrix X and returns a vector of evaluated values.
    """
    if z_score:
        X = zscore(X, axis=1, nan_policy='omit')

    # Remove mean() wrapper if present
    if expr.startswith("mean(") and expr.endswith(")"):
        expr_inner = expr[5:-1]
    else:
        expr_inner = expr

    # Prepend np. to sin, cos, tan to ensure they work with eval
    expr_inner = re.sub(r'\b(sin|cos|tan)\b', r'np.\1', expr_inner)

    # Replace ^ with ** for exponentiation
    expr_inner = expr_inner.replace('^', '**')

    # Find all lag terms
    lags = [int(match.group(1)) for match in re.finditer(r'X_t\+(\d+)', expr_inner)]
    
    min_lag = min(lags) if lags else 0
    max_lag = max(lags) if lags else 0

    expr_converted = re.sub(r'X_t\+(\d+)', r'x[i + \1]', expr_inner)
    #print(expr_converted)

    # Evaluate expression for each row
    result = []
    for row in X:
        T = len(row)
        valid_length = T - max_lag
        
        if valid_length <= 0:
            result.append(np.nan)
            continue

        try:
            x = row
            values = []
            for i in range(valid_length):
                values.append(eval(expr_converted, {"x": x, "i": i, "np": np}))
            result.append(np.nanmean(values))
        except Exception:
            result.append(np.nan)

    return np.array(result)
