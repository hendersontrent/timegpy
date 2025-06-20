from .parsers import tree_to_feature_string
from .evaluate_expression import evaluate_expression

def evaluate_tree(tree, X, z_score=True):
    """
    Evaluate a tree expression on each row (time series) in X using string parsing.
    Returns a 1D numpy array of evaluated values, one per row.
    """
    expr = tree_to_feature_string(tree)
    return evaluate_expression(expr, X, z_score=z_score)
