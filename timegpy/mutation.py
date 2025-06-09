import random
from copy import deepcopy
from .classes import Node
from .generate_random_tree import generate_random_tree
from .get_all_nodes import get_all_nodes
from .replace_node import replace_node

def point_mutation(tree, max_lag=5, max_exponent=4):
    mutated_tree = deepcopy(tree)

    def collect_nodes(node, nodes):
        nodes.append(node)
        if node.left:
            collect_nodes(node.left, nodes)
        if node.right:
            collect_nodes(node.right, nodes)

    # Collect all nodes in the tree
    all_nodes = []
    collect_nodes(mutated_tree, all_nodes)

    # Select a random node
    selected_node = random.choice(all_nodes)

    # Mutate depending on node type
    if selected_node.op is None:  # It's a terminal node
        if selected_node.lag != 0:  # Don't mutate X_t
            selected_node.lag = random.randint(1, max_lag)
        if random.random() < 0.5:  # 50% chance to add/change exponent
            selected_node.exponent = random.randint(2, max_exponent)
        else:
            selected_node.exponent = None
    else:  # It's an operator node
        selected_node.op = random.choice(['+', '-', '*', '/', '^'])

    return mutated_tree

def subtree_mutation(tree: Node, max_lag_terms=4, prob_exponent=0.3, max_lag=5, max_exponent=4) -> Node:
    tree = deepcopy(tree)
    nodes = get_all_nodes(tree, include_root=True)
    target = random.choice(nodes)

    # Mutate with a small random subtree (at least 2 lag terms)
    subtree_lag_terms = random.randint(2, min(max_lag_terms, 4))

    new_subtree = generate_random_tree(
        max_depth=subtree_lag_terms,
        prob_exponent=prob_exponent,
        max_lag=max_lag,
        max_exponent=max_exponent,
        force_X_t=True
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