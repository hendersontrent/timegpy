import pandas as pd
from .evaluate_expression import evaluate

def create(expressions, X, z_score=True):
    """
    Evaluate multiple time-average feature expressions on each row of X and return as a DataFrame.

    Args:
        expressions (list of str): List of expressions in timegpy format.
        X (ndarray): 2D array of time series, rows = series, columns = time points.
        z_score (bool): Whether to z-score each time series before evaluation.

    Returns:
        pd.DataFrame: DataFrame with shape (n_series, n_expressions) containing feature values.
    """
    feature_data = {}

    for expr in expressions:
        feature_data[expr] = evaluate(expr, X, z_score=z_score)

    return pd.DataFrame(feature_data)
