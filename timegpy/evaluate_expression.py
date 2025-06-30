import numpy as np
import re
from scipy.stats import zscore

def evaluate(expr, X, z_score=True):
    if z_score:
        X = zscore(X, axis=1, nan_policy='omit')

    # Remove mean() wrapper if present

    if expr.startswith("mean(") and expr.endswith(")"):
        expr_inner = expr[5:-1]
    else:
        expr_inner = expr

    # Prepend np. to sin, cos, tan to ensure they work with programmtic evaluation

    expr_inner = re.sub(r'\b(sin|cos|tan)\b', r'np.\1', expr_inner)

    # Replace ^ with ** for Pythonic exponentiation

    expr_inner = expr_inner.replace('^', '**')

    # Find all lag terms

    lags = [int(match.group(1)) for match in re.finditer(r'X_t\+(\d+)', expr_inner)]
    max_lag = max(lags) if lags else 0

    # Construct final string expression

    expr_converted = re.sub(r'X_t\+(\d+)', r'x[i + \1]', expr_inner)
    #print(expr_converted)

    # Evaluate expression for each row in X (NOTE: Is there a better way to do this in Python???)

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
