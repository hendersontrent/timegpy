import numpy as np
from .evaluate_tree import evaluate_tree
from .fitness import compute_eta_squared
from .calculate_program_size import calculate_program_size

def evaluate_program_wrapper(args):
    program, X, y = args
    feature = evaluate_tree(program, X)
    if feature is None or np.isnan(feature).all():
        return np.nan, calculate_program_size(program)
    try:
        fitness = compute_eta_squared(feature, y)
    except:
        fitness = np.nan
    size = calculate_program_size(program)
    return fitness, size