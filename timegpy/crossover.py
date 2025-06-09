import random
from copy import deepcopy
from classes import Node
from .replace_node import replace_node
from .get_all_nodes import get_all_nodes

def crossover(parent1: Node, parent2: Node) -> Node:
    p1 = deepcopy(parent1)
    p2 = deepcopy(parent2)

    # Get all nodes including root for crossover
    nodes1 = get_all_nodes(p1, include_root=True)
    nodes2 = get_all_nodes(p2, include_root=True)

    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)

    # Safety: ensure child retains X_t
    def has_X_t(node):
        if node.op is None and node.lag == 0:
            return True
        if node.left and has_X_t(node.left):
            return True
        if node.right and has_X_t(node.right):
            return True
        return False

    def count_X_t(node):
        """Count the number of X_t (lag==0) nodes."""
        count = 0
        def recurse(n):
            nonlocal count
            if n.op is None and n.lag == 0:
                count += 1
            if n.left:
                recurse(n.left)
            if n.right:
                recurse(n.right)
        recurse(node)  # ← this was the bug: changed from recurse(n) to recurse(node)
        return count

    # If crossover_point1 is the only X_t node in parent1, protect it
    if crossover_point1.op is None and crossover_point1.lag == 0:
        if count_X_t(p1) == 1:
            return p1  # Abandon crossover to preserve X_t

    # Perform crossover
    child = deepcopy(p1)
    replace_node(child, crossover_point1, deepcopy(crossover_point2))

    # Final check: ensure child still has X_t
    if not has_X_t(child):
        return p1  # Fallback to original parent

    return child