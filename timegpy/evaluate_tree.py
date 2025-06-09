import numpy as np

def evaluate_tree(tree, X):
    """
    Evaluate a symbolic tree over all rows in X.
    Each row is a time series. Returns a 1D array of feature values (one per row).
    """
    n_samples, T = X.shape

    def eval_node(node, row):
        if node is None:
            return np.full(1, np.nan)

        if node.op is None:
            lag = node.lag
            if lag is None or lag >= T or lag < 0:
                return np.full(1, np.nan)

            val = row[lag:T]
            if len(val) == 0:
                return np.full(1, np.nan)

            if node.exponent is not None:
                try:
                    val = val ** node.exponent
                except Exception:
                    return np.full(len(val), np.nan)

            return val

        # Internal node (operator)
        left = eval_node(node.left, row)
        right = eval_node(node.right, row)

        # If either is None or invalid, return nan array
        if left is None or right is None or not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return np.full(1, np.nan)

        # Ensure same length
        min_len = min(len(left), len(right))
        if min_len == 0:
            return np.full(1, np.nan)

        left = left[:min_len]
        right = right[:min_len]

        try:
            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.true_divide(left, right)
                    result[~np.isfinite(result)] = 0
                    return result
            else:
                return np.full(min_len, np.nan)
        except Exception:
            return np.full(min_len, np.nan)

    # Apply to each row of X
    feature_values = []
    for row in X:
        try:
            result = eval_node(tree, row)
            mean_val = np.nanmean(result)
        except Exception:
            mean_val = np.nan
        feature_values.append(mean_val)

    return np.array(feature_values)