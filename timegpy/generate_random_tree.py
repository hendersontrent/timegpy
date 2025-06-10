import random
from .classes import Node

def generate_random_tree(max_depth, prob_exponent=0.3, max_lag=5, max_exponent=4,
                         force_X_t=True, const_range=(-1.0, 1.0), p_const=0.3):
    """
    Generate a random expression tree with optional float constants.

    Parameters:
    - max_depth: Maximum depth of the tree.
    - prob_exponent: Probability of applying an exponent to a lag node.
    - max_lag: Maximum lag allowed for X_t terms.
    - max_exponent: Maximum value for the exponent.
    - force_X_t: Whether to force the presence of X_t (lag=0) as one of the leaves.
    - const_range: Tuple (low, high) for float constants. If None, constants are disallowed.
    - p_const: Probability of a leaf being a constant instead of a lagged term.
    """

    assert max_depth >= 2, "max_depth must be at least 2"
    assert 0 <= p_const <= 1, "p_const must be between 0 and 1"

    def generate_term(force_lag0=False):
        if not force_lag0 and const_range is not None and random.random() < p_const:
            value = random.uniform(*const_range)
            return Node(value=value)

        lag = 0 if force_lag0 else random.randint(1, max_lag)
        exponent = random.randint(2, max_exponent) if random.random() < prob_exponent else None
        return Node(lag=lag, exponent=exponent)

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
        leaves.append(generate_term(force_lag0=True))  # Force X_t at start
        remaining = n_leaves - 1
    else:
        remaining = n_leaves

    other_terms = [generate_term(force_lag0=False) for _ in range(remaining)]
    random.shuffle(other_terms)

    leaves.extend(other_terms)
    return build_balanced_tree(leaves)
