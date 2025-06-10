from .classes import Node

def tree_to_expression(node):
    if node.is_operator():
        left_str = tree_to_expression(node.left)
        right_str = tree_to_expression(node.right)
        return f"({left_str} {node.op} {right_str})"
    elif node.is_constant():
        return f"{node.value:.4f}"
    elif node.is_lag_term():
        base = f"X_t+{node.lag}"
        if node.exponent is not None:
            return f"({base}**{node.exponent})"
        else:
            return base
    else:
        raise ValueError("Invalid node encountered in tree.")

def tree_to_feature_string(node: Node) -> str:
    return f"mean({tree_to_expression(node)})"