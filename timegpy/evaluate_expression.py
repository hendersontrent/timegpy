import numpy as np
import re

def evaluate_expression(expr, X):
    """
    Evaluates a time-average symbolic expression string (like "mean((X_t+0 * X_t+2))")
    over each row (time series) of the matrix X and returns a vector of evaluated values.
    """
    # 1. Remove mean() wrapper if present
    if expr.startswith("mean(") and expr.endswith(")"):
        expr_inner = expr[5:-1]
    else:
        expr_inner = expr

    # 2. Find all X_t+n terms and determine max lag
    lags = [int(match.group(1)) for match in re.finditer(r'X_t\+(\d+)', expr_inner)]
    max_lag = max(lags) if lags else 0

    # 3. Replace X_t+n with x[i + n]
    expr_converted = re.sub(r'X_t\+(\d+)', r'x[i + \1]', expr_inner)
    #print(expr_converted)

    # 4. Evaluate expression for each row
    result = []
    for row in X:
        T = len(row)
        valid_length = T - max_lag
        if valid_length <= 0:
            result.append(np.nan)
            continue

        try:
            x = row  # the current time series
            values = []
            for i in range(valid_length):
                # safely evaluate the expression
                values.append(eval(expr_converted, {"x": x, "i": i, "np": np}))
            result.append(np.nanmean(values))
        except Exception:
            result.append(np.nan)

    return np.array(result)