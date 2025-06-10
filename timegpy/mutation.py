import random
from copy import deepcopy
from .classes import Node
from .generate_random_tree import generate_random_tree
from .get_all_nodes import get_all_nodes
from .replace_node import replace_node

def point_mutation(tree, max_lag, max_exponent, const_range=(-1.0, 1.0)):
    """
    Mutates a single node in the tree. If const_range is None, no constant mutations occur.
    """
    all_nodes = get_all_nodes(tree)
    node = random.choice(all_nodes)

    if node.is_operator():
        node.op = random.choice(['+', '-', '*', '/'])

    elif node.is_lag_term():
        if node.lag != 0:
            node.lag = random.randint(1, max_lag)
        if random.random() < 0.5:
            node.exponent = random.randint(2, max_exponent)
        else:
            node.exponent = None

    elif node.is_constant() and const_range is not None:
        node.value = random.uniform(*const_range)

    return tree

def subtree_mutation(tree: Node, max_lag_terms=4, prob_exponent=0.3, max_lag=5,
                     max_exponent=4, const_range=(-1.0, 1.0), p_const=0.3) -> Node:
    """
    Perform subtree mutation by replacing a randomly chosen subtree with a new randomly generated one.

    Parameters:
    - tree: The original expression tree.
    - max_lag_terms: Max number of lag/constant terms for the new subtree.
    - prob_exponent: Probability to include exponent on lag nodes.
    - max_lag: Maximum time lag.
    - max_exponent: Maximum exponent value.
    - const_range: Tuple (low, high) or None to control float constant inclusion.
    - p_const: Probability of using a constant node vs a lag node in leaves.
    """
    tree = deepcopy(tree)
    nodes = get_all_nodes(tree, include_root=True)
    target = random.choice(nodes)

    # Generate new subtree
    subtree_lag_terms = random.randint(2, min(max_lag_terms, 4))
    new_subtree = generate_random_tree(
        max_depth=subtree_lag_terms,
        prob_exponent=prob_exponent,
        max_lag=max_lag,
        max_exponent=max_exponent,
        force_X_t=True,
        const_range=const_range,
        p_const=p_const
    )

    if target is tree:
        return new_subtree
    else:
        replace_node(tree, target, new_subtree)
        return tree

def hoist_mutation(tree: Node) -> Node:
    from copy import deepcopy
    tree = deepcopy(tree)

    def collect_subtrees(node):
        """Collects all subtrees (excluding root) that don't remove X_t."""
        subtrees = []
        def recurse(n):
            if n.left:
                subtrees.append(n.left)
                recurse(n.left)
            if n.right:
                subtrees.append(n.right)
                recurse(n.right)
        recurse(node)
        return subtrees

    def count_lag_nodes(node):
        """Counts how many unique lag nodes exist (including X_t)."""
        lags = set()
        def recurse(n):
            if n.op is None:
                lags.add(n.lag)
            if n.left:
                recurse(n.left)
            if n.right:
                recurse(n.right)
        recurse(node)
        return lags

    original_lags = count_lag_nodes(tree)

    if len(original_lags) <= 2:
        return tree  # Don't mutate if only 2 lag values (to keep X_t + something)

    candidates = collect_subtrees(tree)
    random.shuffle(candidates)

    for candidate in candidates:
        lags = count_lag_nodes(candidate)
        if 0 in lags:  # Subtree contains X_t
            return candidate  # Valid hoist mutation

    # If no valid subtree found, return original
    return tree