import random
from .classes import Node

def generate_random_tree(max_lag_terms, prob_exponent=0.3, max_lag=5, max_exponent=4,
                         force_X_t=True, const_range=(-1.0, 1.0), p_const=0.1,
                         p_unary=0.1, unary_set=['sin', 'cos', 'tan']):
    """
    Generate a random expression tree with support for trigonometric functions, limiting number of lag/constant terms.
    """
    assert max_lag_terms >= 1, "max_lag_terms must be at least 1"
    assert 0 <= p_const <= 1, "p_const must be between 0 and 1"
    assert 0 <= p_unary <= 1, "p_unary must be between 0 and 1"

    if unary_set is None or len(unary_set) == 0:
        p_unary = 0.0
    allowed_unary = {'sin', 'cos', 'tan'}
    if not set(unary_set).issubset(allowed_unary):
        raise ValueError("unary_set can only contain 'sin', 'cos', 'tan'")

    def generate_term(force_lag0=False):
        if not force_lag0 and const_range is not None and random.random() < p_const:
            value = random.uniform(*const_range)
            return Node(value=value)
        else:
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

    n_leaves = max(2, max_lag_terms)
    leaves = []

    if force_X_t:
        leaves.append(generate_term(force_lag0=True))
        remaining = n_leaves - 1
    else:
        remaining = n_leaves

    other_terms = [generate_term(force_lag0=False) for _ in range(remaining)]
    random.shuffle(other_terms)
    leaves.extend(other_terms)

    # Randomly apply unary ops to some leaves
    final_leaves = []
    for term in leaves:
        if random.random() < p_unary:
            op = random.choice(unary_set)
            term = Node(op=op, left=term)
        final_leaves.append(term)

    return build_balanced_tree(final_leaves)

