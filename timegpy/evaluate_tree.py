from .parsers import tree_to_feature_string
from .evaluate_expression import evaluate_expression

def evaluate_tree(tree, X, z_score=True):
    expr = tree_to_feature_string(tree)
    return evaluate_expression(expr, X, z_score=z_score)
