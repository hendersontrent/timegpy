import random
from classes import Node

def generate_random_tree(max_depth, prob_exponent=0.3, max_lag=5, max_exponent=4, force_X_t=True):
    """
    Generate a random expression tree with a random number of time-lag terms (leaf nodes)
    between 2 and max_depth. Ensures the X_t (lag = 0) term is always the first leaf if force_X_t=True.
    """
    assert max_depth >= 2, "max_depth must be at least 2"

    def generate_term(force_lag0=False):
        lag = 0 if force_lag0 else random.randint(1, max_lag)
        exponent = random.randint(2, max_exponent) if random.random() < prob_exponent else None
        return Node(op=None, lag=lag, exponent=exponent)

    def build_balanced_tree(terms):
        if len(terms) == 1:
            return terms[0]
        split = random.randint(1, len(terms) - 1)
        left = build_balanced_tree(terms[:split])
        right = build_balanced_tree(terms[split:])
        op = random.choice(['+', '-', '*', '/'])
        return Node(op=op, left=left, right=right)

    # Random number of leaf nodes between 2 and max_depth
    n_leaves = random.randint(2, max_depth)
    leaves = []

    if force_X_t:
        leaves.append(generate_term(force_lag0=True))  # X_t always first
        remaining = n_leaves - 1
    else:
        remaining = n_leaves

    # Fill the rest with random lags
    other_terms = [generate_term(force_lag0=False) for _ in range(remaining)]
    random.shuffle(other_terms)

    leaves.extend(other_terms)  # X_t remains first

    return build_balanced_tree(leaves)